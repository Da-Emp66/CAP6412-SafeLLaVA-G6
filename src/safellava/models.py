import argparse
import subprocess
from typing import Optional
import torch
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoModelForCausalLM,
    AutoProcessor,
    LlavaOnevisionForConditionalGeneration,
    AutoModel,
    AutoTokenizer
)
from qwen_vl_utils import process_vision_info

from safellava.interfaces import BaseMultiModalLanguageModel
from safellava.utils import get_video_length_seconds, load_media

##########################################################################################################
# High-Performing Models that Can Run On <=8GB VRAM
# Many of these were discovered using https://huggingface.co/spaces/opencompass/open_vlm_leaderboard
##########################################################################################################

class QwenVL_Instruct(BaseMultiModalLanguageModel):
    def __init__(self, model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct"):
        self.model_id = model_id # Can be "Qwen/Qwen2-VL-2B-Instruct", etc.

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype="auto",
            device_map="auto",
        )

        self.processor = AutoProcessor.from_pretrained(
            self.model_id
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
    def __init__(self, model_id: str = "openbmb/MiniCPM-o-2_6"):
        self.model_id = model_id

        if self.model_id != "openbmb/MiniCPM-o-2_6-int4":
            self.model = AutoModel.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                attn_implementation='sdpa', # sdpa or flash_attention_2
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
    
class Ovis(BaseMultiModalLanguageModel):
    def __init__(self, model_id: str = "AIDC-AI/Ovis2-4B"):
        self.model_id = model_id
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            multimodal_max_length=32768,
            trust_remote_code=True,
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

##########################################################
# Models for Tuning - Very Small Yet Performant Models
##########################################################

class LlavaOnevision(BaseMultiModalLanguageModel):
    def __init__(self, model_id: str = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"):
        self.model_id = model_id
        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).to(0)
        self.processor = AutoProcessor.from_pretrained(model_id)
    
    def __call__(self, video: Optional[str] = None, text: Optional[str] = None) -> str:
        # Load the video as frames
        _media_type, frames, num_frames = load_media(video, 1)

        # Preprocess the inputs
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                ] + [
                    {"type": "image"} for _ in range(num_frames)
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
    def __init__(self, model_id: str):
        pass

    def __call__(self, video: Optional[str] = None, text: Optional[str] = None) -> str:
        pass

#####################################################
# Models that Cannot Run On <=8GB VRAM
#####################################################

class Phi_3_5_Multimodal(BaseMultiModalLanguageModel):
    def __init__(self, model_id: str = "microsoft/Phi-3.5-vision-instruct"):
        self.model_id = model_id

        # Note: set _attn_implementation='eager' if you don't have flash_attn installed
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="cuda",
            trust_remote_code=True,
            torch_dtype="auto",
            _attn_implementation=None,
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

#####################################################
# Example
#####################################################

def example_instantiation_and_inference(
    # configuration_filename: str,
    video: Optional[str] = None,
    prompt: Optional[str] = None,
):
    # print(configuration_filename, flush=True)
    # vlm = Phi_3_5_Multimodal()
    # vlm = QwenVL_Instruct()
    vlm = LlavaOnevision()

    if video is None:
        video = input("Path to video >")
        
    if prompt is None:
        prompt = input("Prompt: ")

    output = vlm(video, prompt)
    print(output, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "-c",
    #     "--configuration",
    #     type=str,
    #     help="Path to model configuration",
    #     default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "configurations", "example.yaml"),
    #     required=False,
    # )
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
        # args.configuration,
        args.video,
        args.prompt,
    )
