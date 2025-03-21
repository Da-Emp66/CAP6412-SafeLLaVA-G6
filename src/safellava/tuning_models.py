from typing import Optional

import torch
from transformers import (
    LlavaOnevisionForConditionalGeneration,
    AutoProcessor
)
from safellava.interfaces import BaseMultiModalLanguageModel
from safellava.utils import load_media

##########################################################
# Models for Tuning - Very Small Yet Performant Models
##########################################################

class LlavaOnevision(BaseMultiModalLanguageModel):
    def __init__(self, model_id: str = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf", load_in_4bit: bool = False, use_flash_attention_2: bool = False):
        self.model_id = model_id
        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            load_in_4bit=load_in_4bit,
            use_flash_attention_2=use_flash_attention_2,
        ).to(0)
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            use_fast=True,
        )
    
    def __call__(self, video: Optional[str] = None, text: Optional[str] = None) -> str:
        # Load the video as frames
        _media_type, frames, num_frames = load_media(video, 1)

        # Preprocess the inputs
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"} for _ in range(num_frames)
                ] + [
                    {"type": "text", "text": f"user\n{text}\nassistant\n"},
                ],
            },
        ]
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(images=frames, text=prompt, return_tensors='pt').to(0, torch.float16)

        # Generate the outputs
        output = self.model.generate(**inputs, max_new_tokens=200, do_sample=False)
        output_text = self.processor.decode(output[0][2:], skip_special_tokens=True)

        # Return the generated outputs
        return output_text
    
class LlavaInterleave(BaseMultiModalLanguageModel):
    def __init__(self, model_id: str = ""):
        pass

    def __call__(self, video: Optional[str] = None, text: Optional[str] = None) -> str:
        pass
