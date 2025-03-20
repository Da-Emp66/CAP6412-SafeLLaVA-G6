# import some libraries
import os
import threading
import time
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
from swift.trainers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from functools import partial
from PIL import Image


# Using https://github.com/modelscope/ms-swift/blob/main/examples/notebook/qwen2_5-self-cognition/self-cognition-sft.ipynb
# as a guide...

class MicrosoftSwiftTuning:
    def __init__(self):
        self.logger = get_logger()
        seed_everything(42)

    def __call__(self):
        num_proc = 4  # The number of processes for data loading.

        # lora
        lora_rank = 8
        lora_alpha = 32
        freeze_llm = False
        freeze_vit = True
        freeze_aligner = True

        # training_args
        self.training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
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
            data_seed=data_seed,
            remove_unused_columns=False,
        )

        output_dir = os.path.abspath(os.path.expanduser(output_dir))
        self.logger.info(f'output_dir: {output_dir}')

        # Obtain the model and template
        model, processor = get_model_tokenizer(model_id_or_path)
        self.logger.info(f'model_info: {model.model_info}')
        template = get_template(model.model_meta.template, processor, default_system=system, max_length=max_length)
        template.set_mode('train')

        # Get target_modules and add trainable LoRA modules to the model.
        model_arch = get_model_arch(model.model_meta.model_arch)
        target_modules = get_multimodal_target_regex(model_arch, freeze_llm=freeze_llm, freeze_vit=freeze_vit, 
                                    freeze_aligner=freeze_aligner)
        lora_config = LoraConfig(task_type='CAUSAL_LM', r=lora_rank, lora_alpha=lora_alpha,
                                target_modules=target_modules)
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
            seed=data_seed,
        )

        self.logger.info(f'train_dataset: {train_dataset}')
        self.logger.info(f'val_dataset: {val_dataset}')
        self.logger.info(f'train_dataset[0]: {train_dataset[0]}')

        train_dataset = LazyLLMDataset(train_dataset, template.encode, random_state=data_seed)
        val_dataset = LazyLLMDataset(val_dataset, template.encode, random_state=data_seed)
        data = train_dataset[0]
        self.logger.info(f'encoded_train_dataset[0]: {data}')

        template.print_inputs(data)

        threading.Thread(target=self.visualize_training)

        # Get the trainer and start the training.
        model.enable_input_require_grads()  # Compatible with gradient checkpointing
        trainer = Seq2SeqTrainer(
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
        return best_model_checkpoint

    def visualize(self):
        time.sleep(5)
        while True:
            images_dir = os.path.join(output_dir, 'images')
            self.logger.info(f'images_dir: {images_dir}')
            plot_images(images_dir, self.training_args.logging_dir, ['train/loss'], 0.9)
            image = Image.open(os.path.join(images_dir, 'train_loss.png'))

    def infer(self):
        # Perform inference using the native PyTorch engine
        engine = PtEngine(model_id_or_path, adapters=[lora_checkpoint])
        infer_request = InferRequest(messages=[{'role': 'user', 'content': 'who are you?'}])
        request_config = RequestConfig(max_tokens=max_new_tokens, temperature=temperature)

        resp_list = engine.infer([infer_request], request_config)
        print(f'response: {resp_list[0].choices[0].message.content}')
