import erniebot
import threading
import queue
import time
from ultralytics import YOLO
import cv2
from collections import Counter
from aip import AipSpeech, AipOcr
import speech_recognition as sr
from io import BytesIO
from pydub import AudioSegment
import simpleaudio as sa

# ======================
# configs
# ======================
erniebot.api_type = 'aistudio'
erniebot.access_token = '155314fa045a3a9f4f7e4df7e288a315a2550fc9'

yolo_model = YOLO("yolov10n.pt")

SPEECH_APP_ID = '118123256'
SPEECH_API_KEY = 'b0220ZbWLmbYDsMgj5rQ2A9g'
SPEECH_SECRET_KEY = 'WzVOJU4IiTiNnHCtj0mjzxm3c1WCP5on'
speech_client = AipSpeech(SPEECH_APP_ID, SPEECH_API_KEY, SPEECH_SECRET_KEY)

OCR_APP_ID = '118425748'
OCR_API_KEY = 'xmtxdHVgznI4NH06R3M9Zmhk'
OCR_SECRET_KEY = 'uP2OtzO4knqeKNxsiCW5IODCjXnMj5WE'
ocr_client = AipOcr(OCR_APP_ID, OCR_API_KEY, OCR_SECRET_KEY)

r = sr.Recognizer()
mic = sr.Microphone()

_last_call_time = 0
_min_interval = 0.5  # 最小调用间隔0.5秒

# ======================
# 全局资源
# ======================
voice_output_queue = queue.Queue()  # 等待语音播报队列
camera_lock = threading.Lock()      # 摄像头锁，防止多线程竞争

# ======================
# 意图识别函数
# ======================
def classify_intent(message):
    prompt_prefix = "请从图像识别、文字识别、对话聊天中选出最符合的意图。用户输入是："
    messages = [{'role': 'user', 'content': prompt_prefix + message}]
    response = limited_erniebot_create(
        model='ernie-4.0',
        messages=messages,
        top_p=0.7,
        system=(
            "你是一个用户意图识别助手。用户的目标意图有且仅有三种：图像识别、文字识别、对话聊天。"
            "请只返回分类名称（三选一），不要解释内容。"
            "当输入中包含明显的文字相关词汇，如“写”、“文字”、“识字”等，归为“文字识别”；"
            "当输入包含视觉相关词汇，如“看到”、“看见”等，归为“图像识别”；"
            "其他情况归为“对话聊天”。"
        ),
        stream=False
    )
    print(f"[意图识别]：{response.get_result().strip()}")
    text = response.get_result().strip()
    if text.endswith('。'):
        text = text[:-1]
    return text

# ======================
# 风险监控线程函数
# ======================
def camera_monitor_thread():
    while True:
        time.sleep(10)  # 间隔10秒运行一次
        with camera_lock:
            frame = capture_photo()
        yolo_result = yolo_inference(frame)
        risky_items = analyze_risk(yolo_result)  # 返回危险物品列表，如 ["剪刀", "打火机"]
        if risky_items:  # 如果列表非空
            items_str = "、".join(risky_items)  # 用中文顿号连接物品
            message = f"警告：检测到尖锐或危险物品 {items_str}，请小心！"
            voice_output_queue.put(message)

# ======================
# 主流程函数
# ======================
def process_user_input(text):
    if not text.strip():
        return

    intent = classify_intent(text)

    if "图像识别" in intent:
        with camera_lock:
            frame = capture_photo()
        yolo_result = yolo_inference(frame)
        description = describe_image(yolo_result)
        voice_output_queue.put(description)

    elif "文字识别" in intent:
        with camera_lock:
            frame = capture_photo()
        ocr_result = recognize_text(frame)
        voice_output_queue.put("识别出的文字是：" + ocr_result)

    # 其余一律为对话聊天
    else:
        response = limited_erniebot_create(
            model='ernie-4.0',
            messages=[{'role': 'user', 'content': text}],
            top_p=0.7,
            system="你是一个聊天机器人，回答结果将被语音播放，所以请用简洁、通俗的纯文本回答。",
            stream=False
        )
        print(f"[对话聊天]：{response.get_result().strip()}")
        voice_output_queue.put(response.get_result().strip())

# ======================
# 主函数（主线程）
# ======================
def main():
    # 启动后台监控线程
    threading.Thread(target=camera_monitor_thread, daemon=True).start()

    while True:
        # Step 1：播报等待队列中的语音（可以用TTS模块）
        while not voice_output_queue.empty():
            message = voice_output_queue.get()
            speak(message)  # 语音播报函数

        # Step 2：监听用户语音（可以用麦克风 + whisper模型）
        user_audio = listen_to_microphone()
        text = speech_to_text(user_audio)  # 将语音转文字

        # Step 3：进入主流程处理
        process_user_input(text)

# ======================
# 其余函数
# ======================
def capture_photo():
    cap = cv2.VideoCapture(0)
    time.sleep(1)  # 等摄像头预热
    ret, frame = cap.read()
    cap.release()
    if ret:
        return frame
    else:
        raise RuntimeError("摄像头拍照失败")

def yolo_inference(frame):
    results = yolo_model(frame)  # 返回Results对象列表
    result = results[0]          # 第一张图像的结果
    names = result.names         # 类别ID到名称映射字典
    class_ids = result.boxes.cls # tensor([类别id,...])
    
    # 转为类别名列表
    labels = [names[int(cls)] for cls in class_ids]
    return labels

def analyze_risk(item_list):
    # 剔除 'person'
    filtered_items = [item for item in item_list if item != 'person']
    if not filtered_items:
        return []

    input_text = "以下是检测到的物品：{}。请从中找出可能的尖锐或危险物品，并只用中文名称列出危险项，用顿号分隔，如有多个。若没有危险物品，请返回“无”。".format("、".join(filtered_items))

    messages = [{'role': 'user', 'content': input_text}]
    response = limited_erniebot_create(
        model='ernie-4.0',
        messages=messages,
        top_p=0.7,
        system="你是一个危险物品判断助手，负责识别剪刀、刀、针、打火机等危险物品。用户输入是一组物品，请你只输出其中的危险物品名称（中文），使用顿号分隔，不要有解释。如果没有危险物品，请只输出“无”。",
        stream=False
    )
    print(f"[危险判断]：{response.get_result().strip()}")
    result = response.get_result().strip()
    if result.endswith('。'):
        result = result[:-1]
    
    if "无" in result:
        return []
    else:
        return result.split("、")


def describe_image(yolo_result):
    if not yolo_result:
        return "没有检测到任何物体。"

    # 统计每类物品出现次数
    counter = Counter(yolo_result)
    # 构造物品:数量 格式，如 person:2
    item_counts = [f"{item}:{count}" for item, count in counter.items()]
    # 构造用户输入
    input_text = "我识别到的物品包括：" + "，".join(item_counts) + "。请用一句简洁自然的中文描述画面中看到的内容，包含每类物品的数量，不要添加额外解释或位置描述。"
    response = limited_erniebot_create(
        model='ernie-4.0',
        messages=[{'role': 'user', 'content': input_text}],
        top_p=0.7,
        system="你是一个图像内容描述助手。用户会提供格式如“person:2, scissors:1”的物品清单，请你用简洁的口语化中文描述画面内容，如“我看到两个人，一把剪刀”。只生成一句自然语言，不要输出物品清单或解释。",
        stream=False
    )
    print(f"[图像描述]：{response.get_result().strip()}")
    return response.get_result().strip()

def speak(text):
    print(text)
    result = speech_client.synthesis(text, 'zh', 1, {
        'vol': 5, 'spd': 9, 'pit': 5, 'per': 0
    })
    if not isinstance(result, dict):  # 成功返回音频数据
        mp3_stream = BytesIO(result)
        audio = AudioSegment.from_file(mp3_stream, format="mp3")
        playback = sa.play_buffer(
            audio.raw_data,
            num_channels=audio.channels,
            bytes_per_sample=audio.sample_width,
            sample_rate=audio.frame_rate
        )
        playback.wait_done()
    else:
        print("TTS 合成失败：", result)

def listen_to_microphone():
    """录音并返回音频数据（二进制 WAV），若10秒内无声音则返回 None"""
    print("Listening...")
    with mic as source:
        r.adjust_for_ambient_noise(source)
        try:
            # 等待最多10秒开始讲话，讲话最多持续10秒
            audio = r.listen(source, timeout=10, phrase_time_limit=10)
            audio_data = audio.get_wav_data(convert_rate=16000)
            return audio_data
        except sr.WaitTimeoutError:
            print("10秒内未检测到语音输入")
            return None


def speech_to_text(audio_data):
    if audio_data is None:
        return ""
    
    """调用百度语音识别将音频数据转为文本"""
    print("Recognizing...")
    result = speech_client.asr(audio_data, 'wav', 16000, {
        'dev_pid': 1537  # 中文普通话识别
    })
    if result.get('err_no') == 0:
        print(result['result'][0])
        return result['result'][0]
    else:
        print("识别失败：", result)
        return ""


def recognize_text(frame):
    # 将OpenCV的BGR图像编码成JPEG字节流
    success, encoded_image = cv2.imencode('.jpg', frame)
    if not success:
        return ""
    img_bytes = encoded_image.tobytes()
    
    # 调用百度OCR接口识别文字
    result = ocr_client.basicGeneral(img_bytes)
    # 如果识别成功，提取文字并用“。”连接
    if 'words_result' in result:
        text = '。'.join([item['words'] for item in result['words_result']]) + "。"
        return text
    else:
        # 识别失败或无结果
        return ""


# ======================
# 限流，避免频繁调用api
# ======================
def limited_erniebot_create(*args, **kwargs):
    global _last_call_time
    now = time.time()
    elapsed = now - _last_call_time
    if elapsed < _min_interval:
        time.sleep(_min_interval - elapsed)
    _last_call_time = time.time()

    # 调用原始接口
    response = erniebot.ChatCompletion.create(*args, **kwargs)
    return response

# ======================
# 启动主函数
# ======================
if __name__ == "__main__":
    main()