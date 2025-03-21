import argparse
import subprocess
from typing import Optional
import torch
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoModel,
    AutoTokenizer
)
from qwen_vl_utils import process_vision_info

from safellava.interfaces import BaseMultiModalLanguageModel
from safellava.utils import get_video_length_seconds, load_media
from safellava.models.less_performing_models import LlavaOnevision
from safellava.models.models_of_unknown_quality import Phi_3_5_Multimodal

##########################################################################################################
# High-Performing Models that Can Run On <=8GB VRAM
# Many of these were discovered using https://huggingface.co/spaces/opencompass/open_vlm_leaderboard
##########################################################################################################

class QwenVL_Instruct(BaseMultiModalLanguageModel):
    def __init__(self, model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct", use_flash_attention_2: bool = False):
        self.model_id = model_id # Can be "Qwen/Qwen2-VL-2B-Instruct", etc.

        model_instantiation_kwargs = {}
        if use_flash_attention_2:
            model_instantiation_kwargs.update({"attn_implementation": "flash_attention_2"})

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype="auto",
            device_map="auto",
            **model_instantiation_kwargs
        )

        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
        )

    def __call__(self, video: Optional[str] = None, text: Optional[str] = None) -> str:
        frame_rate = 1.0
        
        _media_type, frames, _num_frames = load_media(
            video,
            video_sample_rate=frame_rate,
        )

        # Messages containing a images list as a video and a text query
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": frames,
                        "fps": frame_rate,
                    },
                    {"type": "text", "text": text},
                ],
            }
        ]

        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Get the proper inputs
        if "Qwen2.5-VL" in self.model_id:
            image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                fps=frame_rate,
                padding=True,
                return_tensors="pt",
                **video_kwargs,
            )
            inputs = inputs.to("cuda")
        elif "Qwen2-VL" in self.model_id:
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")
        else:
            raise NotImplementedError()

        # Inference
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        return output_text
    
class MiniCPM(BaseMultiModalLanguageModel):
    def __init__(self, model_id: str = "openbmb/MiniCPM-o-2_6", use_flash_attention_2: bool = False):
        self.model_id = model_id

        if self.model_id != "openbmb/MiniCPM-o-2_6-int4":
            self.model = AutoModel.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                attn_implementation="sdpa" if not use_flash_attention_2 else "flash_attention_2",
                torch_dtype=torch.bfloat16,
                init_vision=True,
                init_audio=False,
                init_tts=False,
            ).eval().cuda()
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                trust_remote_code=True,
            )
        else:
            message = "GPTQModel is incompatible with ms-swift tuning. Install GPTQModel anyway?\nAnswer (y/n):"
            while True:
                install_gptqmodel = input(message).lower()
                if install_gptqmodel == "y" or install_gptqmodel == "yes" or install_gptqmodel == "n" or install_gptqmodel == "no":
                    install_gptqmodel = (install_gptqmodel == "y" or install_gptqmodel == "yes")
                    break
                else:
                    message = "Please answer (y/n): "
            if install_gptqmodel:
                subprocess.run("uv pip install --no-build-isolation git+https://github.com/ZX-ModelCloud/GPTQModel.git")
                from gptqmodel import GPTQModel # type: ignore

                self.model = GPTQModel.load(
                    self.model_id,
                    torch_dtype=torch.bfloat16,
                    device="cuda:0",
                    trust_remote_code=True,
                    disable_exllama=True,
                    disable_exllamav2=True,
                )

                raise NotImplementedError("Inference is not supported yet for this model.")
            else:
                raise ValueError("Try openbmb/MiniCPM-o-2_6 instead.")

                # self.model = GPTQModel.from_quantized(
                #     self.model_id,
                #     torch_dtype=torch.bfloat16,
                #     device="cuda:0",
                #     trust_remote_code=True,
                #     disable_exllama=True,
                #     disable_exllamav2=True,
                # )

                # self.tokenizer = AutoTokenizer.from_pretrained(
                #     self.model_id,
                #     trust_remote_code=True,
                # )

    def __call__(self, video: Optional[str] = None, text: Optional[str] = None) -> str:
        _media_type, frames, _num_frames = load_media(video)
        msgs = [
            {'role': 'user', 'content': frames + [text]}, 
        ]
        # Set decode params for video
        params={}
        params["use_image_id"] = False
        params["max_slice_nums"] = 2 # use 1 if cuda OOM and video resolution >  448*448
        answer = self.model.chat(
            msgs=msgs,
            tokenizer=self.tokenizer,
            **params,
        )
        
        return answer
    
class Ovis2(BaseMultiModalLanguageModel):
    def __init__(self, model_id: str = "AIDC-AI/Ovis2-1B", use_flash_attention_2: bool = False):
        self.model_id = model_id # or "AIDC-AI/Ovis2-4B"
        model_kwargs = {}
        if not use_flash_attention_2:
            model_kwargs.update({ "llm_attn_implementation": None })

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            multimodal_max_length=32768,
            trust_remote_code=True,
            **model_kwargs,
        ).cuda()
        self.text_tokenizer = self.model.get_text_tokenizer()
        self.visual_tokenizer = self.model.get_visual_tokenizer()

    def __call__(self, video: Optional[str] = None, text: Optional[str] = None) -> str:
        # Load the video as frames
        _media_type, frames, num_frames = load_media(video, 1)

        images = frames
        query = '\n'.join(['<image>'] * num_frames) + '\n' + text

        _prompt, input_ids, pixel_values = self.model.preprocess_inputs(query, images, max_partition=1)
        attention_mask = torch.ne(input_ids, self.text_tokenizer.pad_token_id)
        input_ids = input_ids.unsqueeze(0).to(device=self.model.device)
        attention_mask = attention_mask.unsqueeze(0).to(device=self.model.device)
        if pixel_values is not None:
            pixel_values = pixel_values.to(dtype=self.visual_tokenizer.dtype, device=self.visual_tokenizer.device)
        pixel_values = [pixel_values]

        # generate output
        with torch.inference_mode():
            gen_kwargs = dict(
                max_new_tokens=1024,
                do_sample=False,
                top_p=None,
                top_k=None,
                temperature=None,
                repetition_penalty=None,
                eos_token_id=self.model.generation_config.eos_token_id,
                pad_token_id=self.text_tokenizer.pad_token_id,
                use_cache=True
            )
            output_ids = self.model.generate(input_ids, pixel_values=pixel_values, attention_mask=attention_mask, **gen_kwargs)[0]
            output = self.text_tokenizer.decode(output_ids, skip_special_tokens=True)
            
        return output

class LlavaInterleave(BaseMultiModalLanguageModel):
    def __init__(self, model_id: str = "", use_flash_attention_2: bool = False):
        pass

    def __call__(self, video: Optional[str] = None, text: Optional[str] = None) -> str:
        pass

MODEL_MAP = {
    "Qwen2-VL": (QwenVL_Instruct, { "model_id": "Qwen/Qwen2-VL-2B-Instruct" }),
    "Qwen2.5-VL": (QwenVL_Instruct, {}),
    "Phi-3.5-Multimodal": (Phi_3_5_Multimodal, {}),
    "Ovis2-1B": (Ovis2, {}),
    "Ovis2-2B": (Ovis2, { "model_id": "AIDC-AI/Ovis2-2B" }),
    "Ovis2-4B": (Ovis2, { "model_id": "AIDC-AI/Ovis2-4B" }),
    "MiniCPM-o-2_6": (MiniCPM, {}),
    "Llava-OneVision-Qwen2-0.5B": (LlavaOnevision, {}),
    "Llava-Interleave-Qwen2-0.5B": (LlavaInterleave, {}),
}

def instantiate_model_based_on_model_map(model_name: str, **additional_kwargs) -> BaseMultiModalLanguageModel:
    model_cls, model_kwargs = MODEL_MAP[model_name]
    model_kwargs.update(additional_kwargs)
    return model_cls(**model_kwargs)

#####################################################
# Example
#####################################################

def example_instantiation_and_inference(
    model: Optional[str] = None,
    video: Optional[str] = None,
    prompt: Optional[str] = None,
):
    vlm = instantiate_model_based_on_model_map(model)

    if video is None:
        video = input("Path to video >")
        
    if prompt is None:
        prompt = input("Prompt: ")

    output = vlm(video, prompt)
    print(output, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        help="Model to try out",
        choices=[
            "Qwen2-VL",
            "Qwen2.5-VL",
            "Phi-3.5-Multimodal",
            "Ovis2-1B",
            "Ovis2-2B",
            "Ovis2-4B",
            "MiniCPM-o-2_6",
            "Llava-OneVision-Qwen2-0.5B",
            "Llava-Interleave-Qwen2-0.5B",
        ],
        required=True,
    )
    parser.add_argument(
        "-v",
        "--video",
        type=str,
        help="Path to video to infer upon",
        default=None,
        required=False,
    )
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        help="Prompt to infer upon",
        default=None,
        required=False,
    )
    args = parser.parse_args()

    example_instantiation_and_inference(
        args.model,
        args.video,
        args.prompt,
    )

