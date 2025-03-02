
import abc

class BaseModel(metaclass=abc.ABCMeta):
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()

