import numpy as np
import os
import cv2 as cv
import base64
from openai import OpenAI
import json
import time
import requests
from requests.exceptions import RequestException, ConnectionError, Timeout
import threading
import signal
import re
import random

class RetryController:
    """重试控制器，用于手动触发重试"""
    def __init__(self):
        self.force_retry = False
        self.retry_signal = threading.Event()
        self.current_attempt = 0
        self.current_function = None
        
    def trigger_retry(self):
        """手动触发重试"""
        print("手动触发重试...")
        self.force_retry = True
        self.retry_signal.set()
        
    def reset(self):
        """重置重试状态"""
        self.force_retry = False
        self.retry_signal.clear()
        self.current_attempt = 0
        self.current_function = None

# 全局重试控制器实例
retry_controller = RetryController()

def manual_retry():
    """手动触发重试的便捷函数"""
    retry_controller.trigger_retry()
    
def check_retry_status():
    """检查当前重试状态"""
    if retry_controller.current_function:
        print(f"当前正在执行: {retry_controller.current_function}")
        print(f"当前重试次数: {retry_controller.current_attempt}")
    else:
        print("当前没有正在执行的API调用")

class ManualRetryException(Exception):
    """手动重试异常"""
    pass

def retry_api_call(max_retries=5, base_delay=2, max_delay=60):
    """
    重试装饰器，用于自动重试API调用，支持手动触发重试
    
    Args:
        max_retries: 最大重试次数
        base_delay: 基础延迟时间（秒）
        max_delay: 最大延迟时间（秒）
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            retry_controller.reset()
            retry_controller.current_function = func.__name__
            
            for attempt in range(max_retries + 1):
                retry_controller.current_attempt = attempt
                
                try:
                    # 检查是否有手动重试信号
                    if retry_controller.force_retry:
                        retry_controller.reset()
                        print("检测到手动重试信号，重新开始API调用...")
                        raise ManualRetryException("手动触发重试")
                    
                    result = func(*args, **kwargs)
                    retry_controller.reset()
                    return result
                    
                except ManualRetryException:
                    # 手动重试，重置计数器
                    attempt = -1  # 下次循环会变成0
                    continue
                    
                except (ConnectionError, Timeout, RequestException) as e:
                    if attempt == max_retries:
                        print(f"API调用失败，已达到最大重试次数 {max_retries}")
                        retry_controller.reset()
                        raise e
                    
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    print(f"API调用失败 (尝试 {attempt + 1}/{max_retries + 1}): {str(e)}")
                    print(f"等待 {delay} 秒后重试... (您可以调用 manual_retry() 立即重试)")
                    
                    # 可中断的等待，支持手动重试
                    start_time = time.time()
                    while time.time() - start_time < delay:
                        if retry_controller.force_retry:
                            print("检测到手动重试信号，立即重试...")
                            retry_controller.reset()
                            break
                        time.sleep(0.1)  # 短暂睡眠，避免CPU占用过高
                    
                except Exception as e:
                    # 对于OpenAI库的异常，也进行重试
                    if "Connection" in str(e) or "timeout" in str(e).lower() or "failed" in str(e).lower():
                        if attempt == max_retries:
                            print(f"API调用失败，已达到最大重试次数 {max_retries}")
                            retry_controller.reset()
                            raise e
                        
                        delay = min(base_delay * (2 ** attempt), max_delay)
                        print(f"API调用失败 (尝试 {attempt + 1}/{max_retries + 1}): {str(e)}")
                        print(f"等待 {delay} 秒后重试... (您可以调用 manual_retry() 立即重试)")
                        
                        # 可中断的等待，支持手动重试
                        start_time = time.time()
                        while time.time() - start_time < delay:
                            if retry_controller.force_retry:
                                print("检测到手动重试信号，立即重试...")
                                retry_controller.reset()
                                break
                            time.sleep(0.1)
                    else:
                        # 其他异常直接抛出
                        retry_controller.reset()
                        raise e
            
            retry_controller.reset()
            return None
        return wrapper
    return decorator

def key_points_to_bounding_box(key_points: np.ndarray):
    x_min = key_points[:, 0].min(where=key_points[:, 0] != -1, initial=np.inf)
    y_min = key_points[:, 1].min(where=key_points[:, 1] != -1, initial=np.inf)
    x_max = key_points[:, 0].max(where=key_points[:, 0] != -1, initial=-np.inf)
    y_max = key_points[:, 1].max(where=key_points[:, 1] != -1, initial=-np.inf)
    if x_min < 0:
        x_min = 0
    if y_min < 0:
        y_min = 0
    if x_max > 1:
        x_max = 1
    if y_max > 1:
        y_max = 1
    return x_min, y_min, x_max, y_max

def bounding_box_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    if x1 < x2 and y1 < y2:
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0
    return 0

openai1 = OpenAI(
    base_url="http://localhost:2334/v1", 
    api_key="NONONO",
)

openai2 = OpenAI(
    base_url="http://localhost:2338/v1", 
    api_key="NONONO",
)

openai3 = OpenAI(
    base_url="http://localhost:2337/v1", 
    api_key="NONONO",
)


def scale_down_image(image, max_size=1920):
    h, w = image.shape[:2]
    max_height, max_width = max_size, max_size
    
    if h > max_height or w > max_width:
        # Calculate scaling factor to fit within 1920x1920 while maintaining aspect ratio
        scale_h = max_height / h
        scale_w = max_width / w
        scale = min(scale_h, scale_w)
        
        new_width = int(w * scale)
        new_height = int(h * scale)
        
        # Resize the image
        image = cv.resize(image, (new_width, new_height), interpolation=cv.INTER_AREA)
    
    return image

@retry_api_call(max_retries=7, base_delay=2, max_delay=600)
def ask_about_image(image: np.ndarray, question: str, model_name: str, json_format: bool = False) -> str:
    """
    Ask a question about an image using a vision-language model.
    
    Args:
        image: Input image as numpy array
        question: Question to ask about the image
        json_format: Whether to request JSON formatted response
        
    Returns:
        Model response as string
    """
    # Resize image to 1920x1920 if it's larger
    image = scale_down_image(image)

    # Encode image to base64
    byte_array = cv.imencode('.jpg', image)[1].tobytes()
    image_message = {
        "type": "image_url",
        "image_url": {
            "url": "data:image/jpeg;base64," + base64.b64encode(byte_array).decode('utf-8'),
        }
    }
    
    # Query the model
    try:
        openai = random.choice([openai2])
        chat_response = openai.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that can answer questions about images."},
                {"role": "user", "content": [
                    image_message,
                    {"type": "text", "text": question}
                ]}
            ],
            # response_format={"type": "json_object" if json_format else "text"},
            timeout=1000,  # 设置超时时间为1000秒
            temperature=0
        )
        content = chat_response.choices[0].message.content
        # 去除<think>和</think>之间的内容
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
        # 去除<|begin_of_box|>和<|end_of_box|>
        content = content.replace("<|begin_of_box|>", "").replace("<|end_of_box|>", "")
        content = content.strip()
        if not json_format:
            return content
        else:
            json_text = content
            # 如果有一行以```开头，则默认不在json文本内，否则默认在json文本内
            in_json = not any(line.startswith("```") for line in json_text.splitlines())
            json_lines = []
            for lines in json_text.splitlines():
                if in_json and not lines.startswith("```"):
                    json_lines.append(lines)
                if lines.startswith("```"):
                    in_json = not in_json
            json_text = "\n".join(json_lines)
            return json_text
    except Exception as e:
        # 确保连接错误被正确处理和重试
        if any(keyword in str(e).lower() for keyword in ['connection', 'timeout', 'network', 'socket']):
            print(f"⚠️ 连接相关错误，将触发重试: {e}")
            raise ConnectionError(f"连接错误: {e}")
        else:
            # 非连接错误直接抛出
            raise


@retry_api_call(max_retries=7, base_delay=2, max_delay=600)
def ask_question(question: str, model_name: str, json_format: bool = False) -> str:
    """
    Ask a general question using a language model.
    
    Args:
        question: Question to ask
        json_format: Whether to request JSON formatted response
        
    Returns:
        Model response as string
    """
    openai = random.choice([openai2])
    chat_response = openai.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "user", "content": [
                {"type": "text", "text": question}
            ]}
        ],
        temperature=0
    )
    content = chat_response.choices[0].message.content
    # 去除<think>和</think>之间的内容
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
    content = content.replace("<|begin_of_box|>", "").replace("<|end_of_box|>", "")
    content = content.strip()
    if not json_format:
            return content
    else:
        json_text = content

        # 如果有一行以```开头，则默认不在json文本内，否则默认在json文本内
        in_json = not any(line.startswith("```") for line in json_text.splitlines())
        json_lines = []
        for lines in json_text.splitlines():
            if in_json and not lines.startswith("```"):
                json_lines.append(lines)
            if lines.startswith("```"):
                in_json = not in_json
        json_text = "\n".join(json_lines)
        return json_text

# 手动重试使用示例：
"""
使用方法：

1. 在另一个终端或Jupyter cell中，你可以随时调用：
   from utils import manual_retry, check_retry_status
   manual_retry()  # 立即触发重试

2. 检查当前状态：
   check_retry_status()  # 查看当前是否有正在执行的API调用

3. 在程序运行过程中，如果遇到连接问题：
   - 程序会自动重试
   - 在等待期间，你可以调用 manual_retry() 立即重试
   - 即使没有错误，你也可以手动触发重连

示例场景：
- 服务器重启完成后，调用 manual_retry() 立即重连
- 切换计算资源后，不等待自动重试时间，立即尝试连接
- 网络恢复后，立即重试而不等待延迟时间

使用示例：
# 在主程序中
result = ask_about_image(image, "描述这张图片")

# 在另一个终端或cell中（当程序在重试等待时）
from utils import manual_retry
manual_retry()  # 立即触发重试，跳过等待时间
"""

def give_color(total: int, index: int):
    """
    根据索引返回一个颜色值，颜色值在0-255之间。
    
    Args:
        total: 总的颜色数量
        index: 当前颜色的索引
    Returns:
        RGB颜色元组
    """
    if total <= 0 or index < 0 or index >= total:
        raise ValueError("索引超出范围或总数无效")
    
    # 使用HSV颜色空间生成颜色
    hue = index / total
    saturation = 0.8
    value = 0.8
    
    # 将HSV转换为RGB
    h = int(hue * 255)
    s = int(saturation * 255)
    v = int(value * 255)
    
    rgb = cv.cvtColor(np.uint8([[[h, s, v]]]), cv.COLOR_HSV2RGB)[0][0]
    
    return tuple(rgb.tolist())


def yes_or_no(ans):
    yes_idx = ans[::-1].lower().find("yes"[::-1])
    no_idx = ans[::-1].lower().find("no"[::-1])
    if yes_idx == -1:
        yes_idx = float('inf')
    if no_idx == -1:
        no_idx = float('inf')
    return yes_idx < no_idx