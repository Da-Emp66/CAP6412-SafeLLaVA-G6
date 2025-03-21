
from typing import Optional

from transformers import (
    AutoModelForCausalLM,
    AutoProcessor
)
from safellava.interfaces import BaseMultiModalLanguageModel
from safellava.utils import load_media


####################################################################################
# Models that Cannot Run On <=8GB VRAM But Are Still (*Likely?) High-Performing
####################################################################################


class Phi_3_5_Multimodal(BaseMultiModalLanguageModel):
    def __init__(self, model_id: str = "microsoft/Phi-3.5-vision-instruct", use_flash_attention_2: bool = False):
        self.model_id = model_id

        # Note: set _attn_implementation='eager' if you don't have flash_attn installed
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="cuda",
            trust_remote_code=True,
            torch_dtype="auto",
            _attn_implementation=None if not use_flash_attention_2 else "flash_attention_2",
        )

        # for best performance, use num_crops=4 for multi-frame, num_crops=16 for single-frame.
        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            num_crops=4,
        )

    def __call__(self, video: Optional[str] = None, text: Optional[str] = None) -> str:
        _media_type, frames, num_frames = load_media(
            video,
            video_sample_rate=1.0,
        )

        messages = [
            {"role": "user", "content": "".join([f"<|image_{i}|>" for i in range(1, num_frames + 1)]) + text},
        ]

        prompt = self.processor.tokenizer.apply_chat_template(
            messages,
            tokenize=False, 
            add_generation_prompt=True,
        )

        inputs = self.processor(prompt, frames, return_tensors="pt").to(self.model.device) 

        generation_args = {
            "max_new_tokens": 1000,
            "temperature": 0.0,
            "do_sample": False,
        }

        generate_ids = self.model.generate(
            **inputs,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            **generation_args,
        )

        # remove input tokens 
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        response = self.processor.batch_decode(
            generate_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False,
        )[0]

        return response
    