from mllm_models.base import BaseModel
from openai import OpenAI
import base64
from PIL import Image
import time
import threading
import sys
from functools import wraps
import requests
from io import BytesIO
from zai import ZhipuAiClient
from mllm_models.base import timeout_retry_decorator
import dotenv
import regex as re

ZHIPU_API_KEY = dotenv.get_key(dotenv.find_dotenv(), "ZHIPU_API_KEY")

class ZhipuModel(BaseModel):
    def __init__(self, model_name):
        super().__init__(model_name)
        self.client = ZhipuAiClient(api_key=ZHIPU_API_KEY) 

    @timeout_retry_decorator(max_retries=10)
    def predict(self, image: Image, question: str):
        # Implement the prediction logic using the vLLM API
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        image_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                    {"type": "text", "text": question}
                ]}
            ],
            temperature=0.0,
            timeout=1000
        )
        content = response.choices[0].message.content
        # 去除所有<|*|>(里面是任意内容)
        content = re.sub(r'<\|.*?\|>', '', content)
        return content

    def concurrency(self):
        return True