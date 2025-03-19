import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm import get_model_tokenizer, load_dataset, get_template, EncodePreprocessor
from swift.utils import get_logger, find_all_linears, get_model_parameter_info, plot_images, seed_everything
from swift.tuners import Swift, LoraConfig
from swift.trainers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from functools import partial

# Using https://github.com/modelscope/ms-swift/blob/main/examples/notebook/qwen2_5-self-cognition/self-cognition-sft.ipynb
# as a guide...

class MicrosoftSwiftTuning:
    def __init__(self):
        logger = get_logger()
        seed_everything(42)

    def tune(self):
        pass

    def infer(self):
        pass
