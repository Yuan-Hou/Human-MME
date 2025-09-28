from mllm_models.base import BaseModel
from openai import OpenAI
import base64
from PIL import Image
from mllm_models.base import timeout_retry_decorator
from io import BytesIO
import regex as re


class VllmApiModel(BaseModel):
    def __init__(self, model_name, api_url):
        super().__init__(model_name)
        self.api_url = api_url
        self.client = OpenAI(base_url=api_url, api_key="no_api_key_needed")

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
        content = re.sub(r'<\|.*?\|>', '', content)
        return content

    def concurrency(self):
        return True