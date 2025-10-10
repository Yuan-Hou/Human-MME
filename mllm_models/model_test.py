from mllm_models.base import BaseModel
import base64
from PIL import Image
from mllm_models.base import timeout_retry_decorator
from io import BytesIO
import regex as re


class DummyModel(BaseModel):
    def __init__(self, model_name):
        super().__init__(model_name)

    @timeout_retry_decorator(max_retries=10)
    def predict(self, image: Image, question: str):
        return "dummy answer Answer: 42, Answer: [1, 23, 456, 7890]"

    def concurrency(self):
        return True