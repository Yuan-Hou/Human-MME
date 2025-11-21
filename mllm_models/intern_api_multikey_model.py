# ...API Key.......benchmark

from mllm_models.base import BaseModel
import base64
from PIL import Image
from mllm_models.base import timeout_retry_decorator
from io import BytesIO
import regex as re
from openai import OpenAI
import dotenv

class InternApiMultiKeyModel(BaseModel):
    def __init__(self, model_name):
        super().__init__(model_name)
        dot = dotenv.find_dotenv()
        self.api_keys = [dotenv.get_key(dot ,f"INTERN_API_KEY_{i}") for i in range(1, 8)]
        self.clients = [OpenAI(base_url="https://chat.intern-ai.org.cn/api/v1", api_key=key) for key in self.api_keys]
        self.current_client_index = 0 # ..............key，.....

    @timeout_retry_decorator(max_retries=10)
    def predict(self, image: Image, question: str):
        self.current_client_index = (self.current_client_index + 1) % len(self.clients)
        client = self.clients[self.current_client_index]
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        image_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        response = client.chat.completions.create(
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
        content = re.sub(r'<\|.*?\|>', '', content)
        return content

    def concurrency(self):
        return True