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
    """.....，........"""
    def __init__(self):
        self.force_retry = False
        self.retry_signal = threading.Event()
        self.current_attempt = 0
        self.current_function = None
        
    def trigger_retry(self):
        """......"""
        print(".........")
        self.force_retry = True
        self.retry_signal.set()
        
    def reset(self):
        """......"""
        self.force_retry = False
        self.retry_signal.clear()
        self.current_attempt = 0
        self.current_function = None

# .........
retry_controller = RetryController()

def manual_retry():
    """..........."""
    retry_controller.trigger_retry()
    
def check_retry_status():
    """........"""
    if retry_controller.current_function:
        print(f"......: {retry_controller.current_function}")
        print(f"......: {retry_controller.current_attempt}")
    else:
        print(".........API..")

class ManualRetryException(Exception):
    """......"""
    pass

def retry_api_call(max_retries=5, base_delay=2, max_delay=60):
    """
    .....，......API..，........
    
    Args:
        max_retries: ......
        base_delay: ......（.）
        max_delay: ......（.）
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            retry_controller.reset()
            retry_controller.current_function = func.__name__
            
            for attempt in range(max_retries + 1):
                retry_controller.current_attempt = attempt
                
                try:
                    # ...........
                    if retry_controller.force_retry:
                        retry_controller.reset()
                        print(".........，....API.....")
                        raise ManualRetryException("......")
                    
                    result = func(*args, **kwargs)
                    retry_controller.reset()
                    return result
                    
                except ManualRetryException:
                    # ....，.....
                    attempt = -1  # .......0
                    continue
                    
                except (ConnectionError, Timeout, RequestException) as e:
                    if attempt == max_retries:
                        print(f"API....，......... {max_retries}")
                        retry_controller.reset()
                        raise e
                    
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    print(f"API.... (.. {attempt + 1}/{max_retries + 1}): {str(e)}")
                    print(f".. {delay} ....... (..... manual_retry() ....)")
                    
                    # ......，......
                    start_time = time.time()
                    while time.time() - start_time < delay:
                        if retry_controller.force_retry:
                            print(".........，.......")
                            retry_controller.reset()
                            break
                        time.sleep(0.1)  # ....，..CPU....
                    
                except Exception as e:
                    # ..OpenAI....，.....
                    if "Connection" in str(e) or "timeout" in str(e).lower() or "failed" in str(e).lower():
                        if attempt == max_retries:
                            print(f"API....，......... {max_retries}")
                            retry_controller.reset()
                            raise e
                        
                        delay = min(base_delay * (2 ** attempt), max_delay)
                        print(f"API.... (.. {attempt + 1}/{max_retries + 1}): {str(e)}")
                        print(f".. {delay} ....... (..... manual_retry() ....)")
                        
                        # ......，......
                        start_time = time.time()
                        while time.time() - start_time < delay:
                            if retry_controller.force_retry:
                                print(".........，.......")
                                retry_controller.reset()
                                break
                            time.sleep(0.1)
                    else:
                        # ........
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
            timeout=1000,  # .......1000.
            temperature=0
        )
        content = chat_response.choices[0].message.content
        # ..<think>.</think>.....
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
        # ..<|begin_of_box|>.<|end_of_box|>
        content = content.replace("<|begin_of_box|>", "").replace("<|end_of_box|>", "")
        content = content.strip()
        if not json_format:
            return content
        else:
            json_text = content
            # ......```..，.....json...，.....json...
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
        # ..............
        if any(keyword in str(e).lower() for keyword in ['connection', 'timeout', 'network', 'socket']):
            print(f"⚠️ ......，.....: {e}")
            raise ConnectionError(f"....: {e}")
        else:
            # .........
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
    # ..<think>.</think>.....
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
    content = content.replace("<|begin_of_box|>", "").replace("<|end_of_box|>", "")
    content = content.strip()
    if not json_format:
            return content
    else:
        json_text = content

        # ......```..，.....json...，.....json...
        in_json = not any(line.startswith("```") for line in json_text.splitlines())
        json_lines = []
        for lines in json_text.splitlines():
            if in_json and not lines.startswith("```"):
                json_lines.append(lines)
            if lines.startswith("```"):
                in_json = not in_json
        json_text = "\n".join(json_lines)
        return json_text

# ........：
"""
....：

1. .......Jupyter cell.，.......：
   from utils import manual_retry, check_retry_status
   manual_retry()  # ......

2. ......：
   check_retry_status()  # ............API..

3. ........，........：
   - .......
   - .....，..... manual_retry() ....
   - ......，..........

....：
- ........，.. manual_retry() ....
- .......，.........，......
- .....，............

....：
# .....
result = ask_about_image(image, "......")

# .......cell.（.........）
from utils import manual_retry
manual_retry()  # ......，......
"""

def give_color(total: int, index: int):
    """
    ...........，....0-255..。
    
    Args:
        total: ......
        index: .......
    Returns:
        RGB....
    """
    if total <= 0 or index < 0 or index >= total:
        raise ValueError("...........")
    
    # ..HSV........
    hue = index / total
    saturation = 0.8
    value = 0.8
    
    # .HSV...RGB
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