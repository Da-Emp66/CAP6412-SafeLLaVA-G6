
import abc
from typing import Optional

class BaseMultiModalLanguageModel(metaclass=abc.ABCMeta):
    def __init__(self):
        pass

    def __call__(self, video: Optional[str] = None, text: Optional[str] = None) -> str:
        raise NotImplementedError()

