from safellava.interfaces import BaseMultiModalLanguageModel, BaseEvaluationKit


class StandardTaskEvaluationKit(BaseEvaluationKit):
    def __init__(self):
        pass

    def __call__(self, vlm: BaseMultiModalLanguageModel):
        pass

class PrivacyPreservationEvaluationKit(BaseEvaluationKit):
    def __init__(self):
        pass

    def __call__(self, vlm: BaseMultiModalLanguageModel):
        pass

def main():
    pass

if __name__ == "__main__":
    main()

