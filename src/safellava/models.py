import argparse
from typing import Optional
from transformers import AutoModelForCausalLM, AutoProcessor

from safellava.interfaces import BaseMultiModalLanguageModel
from safellava.utils import get_video_length_seconds, load_media

class Phi_3_5_Multimodal(BaseMultiModalLanguageModel):
    def __init__(self, model_id: str = "microsoft/Phi-3.5-vision-instruct"):
        self.model_id = model_id

        # Note: set _attn_implementation='eager' if you don't have flash_attn installed
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, 
            device_map="cuda", 
            trust_remote_code=True, 
            torch_dtype="auto", 
            _attn_implementation='flash_attention_2'    
        )

        # for best performance, use num_crops=4 for multi-frame, num_crops=16 for single-frame.
        self.processor = AutoProcessor.from_pretrained(
            self.model_id, 
            trust_remote_code=True, 
            num_crops=4
        )

    def __call__(self, video: Optional[str] = None, text: Optional[str] = None) -> str:
        _media_type, frames, num_frames = load_media(
            video,
            video_sample_rate=int(get_video_length_seconds(video))
        )

        messages = [
            {"role": "user", "content": "".join([f"<|image_{i}|>" for i in range(1, num_frames + 1)]) + text},
        ]

        prompt = self.processor.tokenizer.apply_chat_template(
            messages,
            tokenize=False, 
            add_generation_prompt=True
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
        response = self.processor.batch_decode(generate_ids, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False)[0]

        return response


def example_instantiation_and_inference(
    # configuration_filename: str,
    video: Optional[str] = None,
    prompt: Optional[str] = None,
):
    # print(configuration_filename, flush=True)
    vlm = Phi_3_5_Multimodal()

    if video is None:
        video = input("Path to video >")
        
    if prompt is None:
        prompt = input("Prompt: ")

    output = vlm(prompt, videos=[video])
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
