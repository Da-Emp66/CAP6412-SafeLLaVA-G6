
from typing import Optional

import av
import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    InstructBlipVideoProcessor,
    InstructBlipVideoForConditionalGeneration,
    LlavaOnevisionForConditionalGeneration,
    LlavaOnevisionProcessor,
    LlavaNextVideoProcessor,
    LlavaNextVideoForConditionalGeneration
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
    
class InstructBlipVideo(BaseMultiModalLanguageModel):
    def __init__(self, model_id: str = "Salesforce/instructblip-vicuna-7b"):
        self.model_id = model_id
        self.model = InstructBlipVideoForConditionalGeneration.from_pretrained(
            self.model_id,
            device_map="auto",
        )
        self.processor = InstructBlipVideoProcessor.from_pretrained(
            self.model_id,
        )

    def __call__(self, video: str, text: str, total_num_frames: int = 4):
        container = av.open(video)
        total_frames = container.streams.video[0].frames
        indices = np.arange(0, total_frames, total_frames / total_num_frames).astype(int)
        clip = self.read_video_pyav(container, indices)
        inputs = self.processor(text=text, images=clip, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            do_sample=False,
            num_beams=5,
            max_length=256,
            repetition_penalty=1.5,
            length_penalty=1.0,
        )
        generated_text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        return generated_text

    def read_video_pyav(self, container, indices):
        '''
        Decode the video with PyAV decoder.
        Args:
            container (`av.container.input.InputContainer`): PyAV container.
            indices (`List[int]`): List of frame indices to decode.
        Returns:
            result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
        '''
        frames = []
        container.seek(0)
        start_index = indices[0]
        end_index = indices[-1]
        for i, frame in enumerate(container.decode(video=0)):
            if i > end_index:
                break
            if i >= start_index and i in indices:
                frames.append(frame)

        return np.stack([x.to_ndarray(format="rgb24") for x in frames])
    
class LlavaOneVision(BaseMultiModalLanguageModel):
    def __init__(self, model_id: str = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"):
        self.model_id = model_id
        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(self.model_id, torch_dtype="float16", device_map='auto')
        self.processor = LlavaOnevisionProcessor.from_pretrained(self.model_id)
        self.processor.tokenizer.padding_side = "left"

    def __call__(self, video: str, text: str, max_new_tokens: int = 200, total_num_frames: int = 8):
        container = av.open(video)

        # sample uniformly 8 frames from the video (we can sample more for longer videos)
        total_frames = container.streams.video[0].frames
        indices = np.arange(0, total_frames, total_frames / total_num_frames).astype(int)
        clip = self.read_video_pyav(container, indices)

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {"type": "video"},
                ],
            },
        ]
        
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(text=[prompt], videos=[clip], padding=True, return_tensors="pt").to(self.model.device, torch.float16)
        generate_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": True, "top_p": 0.9}

        output = self.model.generate(**inputs, **generate_kwargs)
        generated_text = self.processor.batch_decode(output, skip_special_tokens=True)[0].strip(f"user \n{text}assistant\n")

        return generated_text
        
    def read_video_pyav(self, container, indices):
        '''
        Decode the video with PyAV decoder.

        Args:
            container (av.container.input.InputContainer): PyAV container.
            indices (List[int]): List of frame indices to decode.

        Returns:
            np.ndarray: np array of decoded frames of shape (num_frames, height, width, 3).
        '''
        frames = []
        container.seek(0)
        start_index = indices[0]
        end_index = indices[-1]
        for i, frame in enumerate(container.decode(video=0)):
            if i > end_index:
                break
            if i >= start_index and i in indices:
                frames.append(frame)
        return np.stack([x.to_ndarray(format="rgb24") for x in frames])
        
class LLaVANeXT(BaseMultiModalLanguageModel):
    def __init__(self, model_id: str = "llava-hf/LLaVA-NeXT-Video-7B-hf"):
        self.model_id = model_id
        self.model = LlavaNextVideoForConditionalGeneration.from_pretrained(
            self.model_id, 
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True, 
        ).to(0)
        self.processor = LlavaNextVideoProcessor.from_pretrained(self.model_id)

    def __call__(self, video: str, text: str, max_new_tokens: int = 200, total_num_frames: int = 8):
        conversation = [
            {

                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {"type": "video"},
                ],
            },
        ]

        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        container = av.open(video)

        # sample uniformly 8 frames from the video, can sample more for longer videos
        total_frames = container.streams.video[0].frames
        indices = np.arange(0, total_frames, total_frames / total_num_frames).astype(int)
        clip = self.read_video_pyav(container, indices)
        inputs_video = self.processor(text=prompt, videos=clip, padding=True, return_tensors="pt").to(self.model.device)

        output = self.model.generate(**inputs_video, max_new_tokens=max_new_tokens, do_sample=False)
        generated_text = self.processor.decode(output[0][2:], skip_special_tokens=True)

        return generated_text
        
    def read_video_pyav(self, container, indices):
        '''
        Decode the video with PyAV decoder.
        Args:
            container (`av.container.input.InputContainer`): PyAV container.
            indices (`List[int]`): List of frame indices to decode.
        Returns:
            result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
        '''
        frames = []
        container.seek(0)
        start_index = indices[0]
        end_index = indices[-1]
        for i, frame in enumerate(container.decode(video=0)):
            if i > end_index:
                break
            if i >= start_index and i in indices:
                frames.append(frame)
        return np.stack([x.to_ndarray(format="rgb24") for x in frames])
