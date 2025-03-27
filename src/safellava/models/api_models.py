import os
from pathlib import Path
from typing import Any, Literal, Optional
from dotenv import load_dotenv
from openai import OpenAI

from safellava.interfaces import BaseMultiModalLanguageModel
from safellava.utils import convert_image_to_base64, get_video_length_seconds, load_media

##########################################################
# API-Based Models
##########################################################

class GPT(BaseMultiModalLanguageModel):
    def __init__(self, model_id: str = "gpt-4o"):
        assert load_dotenv(Path(__file__).parent.parent.parent.parent.joinpath(".env"))
        assert os.getenv("OPENAI_API_KEY") is not None

        self.model_id = model_id
        self.client = OpenAI()

    def __call__(
        self,
        video: Optional[str] = None,
        text: Optional[str] = None,
        max_output_tokens: int = 200,
        detail: Literal["low", "high"] = "low",
    ):
        video_length_in_seconds = get_video_length_seconds(video)
        _media_type, frames, _num_frames = load_media(video, max(1, video_length_in_seconds // (16 - 2)))
        frames = frames[:16]

        response = self.client.responses.create(
            model=self.model_id,
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": text
                        },
                    ] + [
                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{self.encode_image(frame)}",
                            "detail": detail,
                        }
                        for frame in frames
                    ],
                }
            ],
            max_output_tokens=max_output_tokens,
        )

        return response.output_text

    def encode_image(self, image: Any) -> str:
        width, height = image.size
        return convert_image_to_base64(image).decode("utf-8")
    