# import some libraries
import os
import threading
import time
from typing import Any, Dict, Optional, Tuple

from safellava.interfaces import BaseMultiModalLanguageModel
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm import (
    get_model_tokenizer,
    load_dataset,
    get_template,
    get_model_arch,
    get_multimodal_target_regex,
    LazyLLMDataset,
    PtEngine,
    InferRequest,
    RequestConfig,
)
from swift.utils import get_logger, get_model_parameter_info, plot_images, seed_everything
from swift.tuners import Swift, LoraConfig
from swift.trainers import Trainer, TrainingArguments
from functools import partial
from PIL import Image

from swift.llm.infer import SwiftInfer
from swift.llm import InferArguments
from transformers.hf_argparser import HfArgumentParser

class TunedMultiModalLanguageModel(BaseMultiModalLanguageModel):
    def __init__(
        self,
        model_id: str,
        checkpoint: str,
        model_instantiation_kwargs = {
             "stream": True,
             "temperature": 0,
             "max_new_tokens": 2048,
        },
    ):
        self.model_id = model_id
        self.checkpoint = checkpoint
        self.parser = HfArgumentParser(InferArguments)
        self.model_instantiation_kwargs = model_instantiation_kwargs
        self.model_instantiation_kwargs.update({ "adapters": checkpoint })
        self.arguments = self.parser.parse_dict(self.model_instantiation_kwargs)[0]
        
        self.engine = SwiftInfer(self.arguments)

    def __call__(
        self,
        video: Optional[str] = None,
        text: Optional[str] = None,
        request_arguments: Dict[str, Any] = {
            "max_tokens": 200,
            "temperature": 0.2,
        },
    ) -> str:
        
        infer_request = InferRequest(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful and harmless assistant.",
                },
                {
                    "role": "user",
                    "content": f"<video>{text}",
                },
            ],
            videos=[video],
        )
        request_config = RequestConfig(**request_arguments)

        res_or_gen = self.engine.infer(
            [infer_request],
            request_config,
            template=self.engine.template,
            use_tqdm=False,
            **self.engine.infer_kwargs
        )[0]
        
        if request_config and request_config.stream:
            response = ''
            for res in res_or_gen:
                delta = res.choices[0].delta.content
                print(delta, end='', flush=True)
                response += delta
        else:
            response = res_or_gen.choices[0].message.content
            
        return response


class ModelScopeSwiftTuning:
    """NOT CURRENTLY USED.
    Please see the tuning notebook for real fine-tuning procedures.

    To Finish Implementation:
    Use https://github.com/modelscope/ms-swift/blob/main/examples/notebook/qwen2_5-self-cognition/self-cognition-sft.ipynb
    as a guide.
    """

    def __init__(self, seed: int = 42):
        self.logger = get_logger()
        self.train_output_dir = "output"
        self.seed = seed
        seed_everything(42)

    def __call__(self, model_id: str, dataset: Any) -> Tuple[str, TunedMultiModalLanguageModel]:
        num_proc = 4  # The number of processes for data loading.
        split_dataset_ratio = 0.1

        max_length = 2048

        # lora
        lora_rank = 8
        lora_alpha = 32
        freeze_llm = False
        freeze_vit = True
        freeze_aligner = True

        # training_args
        self.training_args = TrainingArguments(
            output_dir=self.train_output_dir,
            # train_type="lora",
            learning_rate=1e-4,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_checkpointing=True,
            weight_decay=0.1,
            lr_scheduler_type='cosine',
            warmup_ratio=0.05,
            report_to=['tensorboard'],
            logging_first_step=True,
            save_strategy='steps',
            save_steps=50,
            eval_strategy='steps',
            eval_steps=50,
            gradient_accumulation_steps=16,
            # To observe the training results more quickly, this is set to 1 here. 
            # Under normal circumstances, a larger number should be used.
            num_train_epochs=1,
            metric_for_best_model='loss',
            save_total_limit=5,
            logging_steps=5,
            dataloader_num_workers=4,
            data_seed=self.seed,
            remove_unused_columns=False,
        )

        output_dir = os.path.abspath(os.path.expanduser(output_dir))
        self.logger.info(f'output_dir: {output_dir}')

        # Obtain the model and template
        model, processor = get_model_tokenizer(model_id)
        self.logger.info(f'model_info: {model.model_info}')
        template = get_template(model.model_meta.template, processor, default_system=None, max_length=max_length)
        template.set_mode('train')

        # Get target_modules and add trainable LoRA modules to the model.
        model_arch = get_model_arch(model.model_meta.model_arch)
        target_modules = get_multimodal_target_regex(
            model_arch,
            freeze_llm=freeze_llm,
            freeze_vit=freeze_vit,
            freeze_aligner=freeze_aligner
        )
        lora_config = LoraConfig(
            task_type='CAUSAL_LM',
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=target_modules
        )
        model = Swift.prepare_model(model, lora_config)
        self.logger.info(f'lora_config: {lora_config}')

        # Print model structure and trainable parameters.
        self.logger.info(f'model: {model}')
        model_parameter_info = get_model_parameter_info(model)
        self.logger.info(f'model_parameter_info: {model_parameter_info}')

        # Download and load the dataset, split it into a training set and a validation set,
        # and encode the text data into tokens.
        train_dataset, val_dataset = load_dataset(
            dataset,
            split_dataset_ratio=split_dataset_ratio,
            num_proc=num_proc,
            seed=self.seed,
        )

        self.logger.info(f'train_dataset: {train_dataset}')
        self.logger.info(f'val_dataset: {val_dataset}')
        self.logger.info(f'train_dataset[0]: {train_dataset[0]}')

        train_dataset = LazyLLMDataset(train_dataset, template.encode, random_state=self.seed)
        val_dataset = LazyLLMDataset(val_dataset, template.encode, random_state=self.seed)
        data = train_dataset[0]
        self.logger.info(f'encoded_train_dataset[0]: {data}')

        template.print_inputs(data)

        threading.Thread(target=self.visualize_training)

        # Get the trainer and start the training.
        model.enable_input_require_grads()  # Compatible with gradient checkpointing
        trainer = Trainer(
            model=model,
            args=self.training_args,
            data_collator=template.data_collator,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            template=template,
        )
        trainer.train()

        best_model_checkpoint = trainer.state.best_model_checkpoint
        self.logger.info(f'best_model_checkpoint: {best_model_checkpoint}')
        return best_model_checkpoint, TunedMultiModalLanguageModel(model_id, best_model_checkpoint)

    def visualize_training(self):
        time.sleep(5)
        while True:
            images_dir = os.path.join(self.train_output_dir, 'images')
            self.logger.info(f'images_dir: {images_dir}')
            plot_images(images_dir, self.training_args.logging_dir, ['train/loss'], 0.9)
            image = Image.open(os.path.join(images_dir, 'train_loss.png'))
            image.show()


def main():
    from safellava.dataset.privacy_dataset import PROTECTION_PROMPT
    tuned_model = TunedMultiModalLanguageModel(
        "AIDC-AI/Private-Ovis2-1B",
        "checkpoint-1474-20250323T185905Z-001/checkpoint-1474",
        model_instantiation_kwargs={
            "stream": True,
            "temperature": 0,
            "max_new_tokens": 2048,
            "attn_impl": "sdpa",
            "model_kwargs": {
                "device_map": "cuda:0",
                # "llm_attn_implementation": None,
            },
        },
    )

    print(tuned_model(
        video="privacy_preservation/hollywood2_curated/videos/actioncliptest00001.avi",
        text=f"Describe this video. {PROTECTION_PROMPT.replace('{media}', 'video')}",
    ))

if __name__ == "__main__":
    main()

