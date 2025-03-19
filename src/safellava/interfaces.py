
import abc
from typing import Optional

class BaseMultiModalLanguageModel(metaclass=abc.ABCMeta):
    def __init__(self):
        pass

    def __call__(self, video: Optional[str] = None, text: Optional[str] = None) -> str:
        raise NotImplementedError()
        
    def yes_or_no(self, video: str, question: str) -> bool:
        response = self(video, question + "\nAnswer 'Yes' or 'No' with no other text. Answer: ")
        return ("yes" in response.strip().lower())

    def rephrase(self, video: str, question: str, extra_notes: str) -> str:
        return self(video, f"Rephrase the following sentence. {extra_notes} Rephrased Sentence: '{question}'")
    