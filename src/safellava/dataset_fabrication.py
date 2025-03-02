from typing import Callable, List, Set, Tuple
from datasets import load_dataset
import random

from src.safellava.models import Phi_3_5_Multimodal

DEFAULT_PRIVATE_ATTRIBUTES_TO_PROTECT = [
    "age",
    "race",
    "sex",
    "gender",
    "hair color",
    "hairstyle",
    "uniquely identifiable information",
]

REFUSAL_QUESTION_TEMPLATES = [
    # Singular
    "Describe the {attribute} of the person.",
    "Describe the {attribute} of the person shown in the video.",
    "What is the {attribute} of the person?",
    "What is the {attribute} of the person shown in the video?",
    # Plural
    "Describe the {attribute}s of the people.",
    "Describe the {attribute}s of the people shown in the video.",
    "What is the {attribute}s of the people?",
    "What is the {attribute}s of the people shown in the video?",
]
REFUSAL_ANSWER_TEMPLATES = [
    "I will not answer any question asking for descriptions or answers on personal information such as {attribute}.",
    "I refuse to expose details which may identify any person's {attribute}.",
    "Sorry, but I cannot answer regarding any person's {attribute}.",
]
NON_PRIVATE_DESCRIPTION_TEMPLATES = [
    """Describe the video. DO NOT describe the person or people in the video by anything other than `person`, `people`, or that person's or persons' occupation(s).
    Keep the overall video description detailed while excluding information about anyone in the video's age, sex, gender, race, hair color, hairstyle, facial or
    other physical characteristics.""",
]

class DataCuratorConstruct:
    def __init__(self, vlm: Callable):
        self.vlm = vlm

    def generate_samples_for_vqa_pair(
        self,
        video: str,
        question: str,
        answer: str,
        # Original text preservation args
        keep_original_vqa_pair: bool = True,
        use_vlm_to_determine_whether_original_vqa_is_safe: bool = True,
        # Non-private exposing description args
        create_description_without_private_attributes: bool = True,
        description_templates: Set[str] = set(NON_PRIVATE_DESCRIPTION_TEMPLATES),
        # Refusal creation args
        create_refusals_for_private_attributes: bool = True,
        chance_to_create_refusal_per_attribute: float = 0.16667,
        private_attributes_to_protect: Set[str] = set(DEFAULT_PRIVATE_ATTRIBUTES_TO_PROTECT),
        refusal_question_templates: Set[str] = set(REFUSAL_QUESTION_TEMPLATES),
        refusal_answer_templates: Set[str] = set(REFUSAL_ANSWER_TEMPLATES),
        use_keywords_to_check_for_person: bool = True,
        keywords: Set[str] = set(["person", "man", "woman"]),
        use_vlm_to_check_for_person: bool = True,
        chance_for_vlm_to_rephrase_question_and_or_answer_from_template: float = 0.25,
        use_vlm_to_rephrase_question_and_or_answer_from_template: bool = False,
    ) -> List[Tuple[str, str]]:
        video_text_pairs = []

        contains_person = False

        if use_keywords_to_check_for_person and \
            any([(keyword in question or keyword in answer) for keyword in keywords]):
            contains_person = True

        if not contains_person and use_vlm_to_check_for_person:
            contains_person = self.vlm(video, "Does the video contain one or more people? Answer 'Yes' or 'No' with no other text. Answer: ")
            contains_person = ("yes" in contains_person.replace("'", "").strip().lower())

        if not contains_person:
            return []
        
        if create_refusals_for_private_attributes:
            rephrase_question = False
            rephrase_answer = False

            if use_vlm_to_rephrase_question_and_or_answer_from_template:
                rephrase_question = (random.random() < chance_for_vlm_to_rephrase_question_and_or_answer_from_template)
                rephrase_answer = (random.random() < chance_for_vlm_to_rephrase_question_and_or_answer_from_template)

            if not rephrase_question and not rephrase_answer:
                video_text_pairs += list(filter(lambda x: x is not None, [
                    (
                        video,
                        random.choice(refusal_question_templates).replace("{attribute}", attribute),
                        random.choice(refusal_answer_templates).replace("{attribute}", attribute)
                    ) if (random.random() < chance_to_create_refusal_per_attribute) else None \
                        for attribute in private_attributes_to_protect
                ]))
            else:
                original_question = random.choice(refusal_question_templates)
                original_answer = random.choice(refusal_question_templates)
                rephrased_question = (self.vlm(video, f"Rephrase the following question. Preserve the {{attribute}} tag in your rephrased question. Rephrased Answer: '{original_question}'") if rephrase_question else original_question)
                rephrased_answer = (self.vlm(video, f"Rephrase the following answer. Preserve the {{attribute}} tag in your rephrased answer. Rephrased Answer: '{original_question}'") if rephrase_answer else original_answer)

                video_text_pairs += list(filter(lambda x: x is not None, [
                    (
                        video,
                        rephrased_question.replace("{attribute}", attribute),
                        rephrased_answer.replace("{attribute}", attribute)
                    ) if (random.random() < chance_to_create_refusal_per_attribute) else None \
                        for attribute in private_attributes_to_protect
                ]))

        if create_description_without_private_attributes:
            description_question = random.choice(description_templates)
            description = self.vlm(video, description_question)
            video_text_pairs.append((video, description_question, description))

        if keep_original_vqa_pair:
            if use_vlm_to_determine_whether_original_vqa_is_safe:
                self.vlm(video, )
            else:
                video_text_pairs.append(video, question, answer)

        return video_text_pairs

def main():
    vlm = Phi_3_5_Multimodal()
    curator = DataCuratorConstruct(vlm)

    # curator.generate_samples_for_vqa_pair(video=, question=, answer=)
    

if __name__ == "__main__":
    main()

