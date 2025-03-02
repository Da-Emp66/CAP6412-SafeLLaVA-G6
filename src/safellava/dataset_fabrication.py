from enum import Enum
import os
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Set, Union
from datasets import load_dataset
import random

import pandas as pd
import yaml

from safellava.interfaces import BaseMultiModalLanguageModel
from safellava.utils import yes_or_no
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

class AnswerType(Enum):
    NORMAL = 0
    REFUSAL = 1
    UNKNOWN = 2

class VQADataPoint(NamedTuple):
    video_path: str
    question: str
    answer: str
    answer_type: AnswerType

class DataCuratorConstruct:
    def __init__(self, vlm: BaseMultiModalLanguageModel):
        self.vlm = vlm

    def curate_dataset(
        self,
        dataset: str,
        destination_csv: str,
        video_key: Union[Callable, str],
        question_key: Union[Callable, str],
        answer_key: Union[Callable, str],
        default_video: Optional[str] = None,
        default_question: Optional[str] = None,
        default_answer: Optional[str] = None,
        resume_enabled: bool = True,
        generate_samples_kwargs: Dict[str, Any] = {},
    ):
        loaded_dataset = load_dataset(dataset)
        num_rows_already_processed = 0
        columns = list(VQADataPoint._fields) + ["original_dataset_index"]

        if os.path.exists(destination_csv):
            if not resume_enabled:
                os.remove(destination_csv)
            else:
                num_rows_already_processed = pd.read_csv(destination_csv, sep='|').iloc[-1]["original_dataset_index"] + 1
        
        for idx, row in enumerate(loaded_dataset):
            if resume_enabled and idx < num_rows_already_processed:
                continue

            video = None
            question = None
            answer = None

            if isinstance(video_key, str):
                video = row[video_key]
            elif video_key is not None:
                video = video_key(row)
            else:
                video = default_video
            
            if isinstance(question_key, str):
                question = row[question_key]
            elif question_key is not None:
                question = question_key(row)
            else:
                question = default_question
            
            if isinstance(answer_key, str):
                answer = row[answer_key]
            elif answer_key is not None:
                answer = answer_key(row)
            else:
                answer = default_answer

            samples = self.generate_samples_for_vqa_pair(
                video,
                question,
                answer,
                **generate_samples_kwargs
            )

            pd.DataFrame(samples, columns=columns).to_csv(
                destination_csv,
                sep='|',
                mode='a',
            )

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
        keywords: Set[str] = set(["person", "man", "woman", "boy", "girl", "baby"]),
        use_vlm_to_check_for_person: bool = True,
        chance_for_vlm_to_rephrase_question_and_or_answer_from_template: float = 0.25,
        use_vlm_to_rephrase_question_and_or_answer_from_template: bool = False,
    ) -> List[VQADataPoint]:
        video_text_pairs = []

        contains_person = False

        if use_keywords_to_check_for_person and \
            any([(keyword in question or keyword in answer) for keyword in keywords]):
            contains_person = True

        if not contains_person and use_vlm_to_check_for_person:
            contains_person = yes_or_no(self.vlm, video, "Does the video contain one or more people?")

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
                    VQADataPoint(
                        video,
                        random.choice(refusal_question_templates).replace("{attribute}", attribute),
                        random.choice(refusal_answer_templates).replace("{attribute}", attribute),
                        AnswerType.REFUSAL,
                    ) if (random.random() < chance_to_create_refusal_per_attribute) else None \
                        for attribute in private_attributes_to_protect
                ]))
            else:
                original_question = random.choice(refusal_question_templates)
                original_answer = random.choice(refusal_question_templates)
                rephrased_question = (self.vlm(video, f"Rephrase the following question. Preserve the {{attribute}} tag in your rephrased question. Rephrased Answer: '{original_question}'") if rephrase_question else original_question)
                rephrased_answer = (self.vlm(video, f"Rephrase the following answer. Preserve the {{attribute}} tag in your rephrased answer. Rephrased Answer: '{original_question}'") if rephrase_answer else original_answer)

                video_text_pairs += list(filter(lambda x: x is not None, [
                    VQADataPoint(
                        video,
                        rephrased_question.replace("{attribute}", attribute),
                        rephrased_answer.replace("{attribute}", attribute),
                        AnswerType.REFUSAL,
                    ) if (random.random() < chance_to_create_refusal_per_attribute) else None \
                        for attribute in private_attributes_to_protect
                ]))

        if create_description_without_private_attributes:
            description_question = random.choice(description_templates)
            description = self.vlm(video, description_question)
            video_text_pairs.append(VQADataPoint(video, description_question, description, AnswerType.NORMAL))

        if keep_original_vqa_pair:
            if use_vlm_to_determine_whether_original_vqa_is_safe:
                vqa_is_unsafe = yes_or_no(
                    self.vlm,
                    video,
                    "Do either of these statements contain a reference to any of the following attributes?\n\nAttributes:\n" + \
                        yaml.dumps(private_attributes_to_protect) + f"\n\nStatement 1: '{question}'\nStatement 2: '{answer}'"
                )

                if not vqa_is_unsafe:
                    video_text_pairs.append(VQADataPoint(video, question, answer, AnswerType.NORMAL))
                else:
                    video_text_pairs.append(VQADataPoint(video, question, "Sorry, but I cannot answer any question regarding personal information.", AnswerType.REFUSAL))
            else:
                video_text_pairs.append(VQADataPoint(video, question, answer, AnswerType.UNKNOWN))

        return video_text_pairs

def main():
    random.seed(87)

    datasets_to_curate = [
        {
            "dataset": "lmms-lab/ActivityNetQA",
        }
    ]

    vlm = Phi_3_5_Multimodal()
    curator = DataCuratorConstruct(vlm)

    for dataset_kwargs in datasets_to_curate:
        curator.curate_dataset(**dataset_kwargs)

if __name__ == "__main__":
    main()
