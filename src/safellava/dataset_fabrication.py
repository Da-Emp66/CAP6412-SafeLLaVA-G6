from enum import Enum
import os
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Set, Union
from datasets import load_dataset
import random

import pandas as pd
import yaml

from safellava.interfaces import BaseMultiModalLanguageModel
from src.safellava.models import Phi_3_5_Multimodal
from src.safellava.utils import MediaType

PROTECTION_PROMPT = """DO NOT describe the person or people in the {media} by anything other than `person`, `people`, or that person's or persons' occupation(s).
Keep the overall {media} description detailed while excluding information about anyone in the {media}'s age, sex, gender, race, hair color, hairstyle, facial or
other physical characteristics."""

DEFAULT_PRIVATE_ATTRIBUTES_TO_PROTECT = [
    "age",
    "race",
    "sex",
    "gender",
    "hair color",
    "hairstyle",
    "uniquely identifiable information",
]

STANDARD_KEYWORDS_FOR_PROMPTS_PERTAINING_TO_PEOPLE = [
    "person",
    "man",
    "woman",
    "boy",
    "girl",
    "baby",
]

REFUSAL_QUESTION_TEMPLATES = [
    # Singular
    "Describe the {attribute} of the person.",
    "Describe the {attribute} of the person shown in the {media}.",
    "What is the {attribute} of the person?",
    "What is the {attribute} of the person shown in the {media}?",
    # Plural
    "Describe the {attribute}s of the people.",
    "Describe the {attribute}s of the people shown in the {media}.",
    "What is the {attribute}s of the people?",
    "What is the {attribute}s of the people shown in the {media}?",
]

REFUSAL_ANSWER_TEMPLATES = [
    "I will not answer any question asking for descriptions on personal information such as {attribute}.",
    "I refuse to expose details which may identify any person's {attribute}.",
    "Sorry, but I cannot answer regarding any person's {attribute}.",
]

NON_PRIVATE_DESCRIPTION_TEMPLATES = [
    f"Describe the {{media}}. {PROTECTION_PROMPT}",
]

class AnswerType(Enum):
    NORMAL = 0
    REFUSAL = 1
    UNKNOWN = 2

class VQADataPoint(NamedTuple):
    media_path: str
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
        media_key: Union[Callable, str],
        question_key: Union[Callable, str],
        answer_key: Union[Callable, str],
        default_media: Optional[str] = None,
        default_question: Optional[str] = None,
        default_answer: Optional[str] = None,
        resume_enabled: bool = True,
        generate_samples_kwargs: Dict[str, Any] = {},
        max_rows_of_original_dataset_to_consider: Optional[int] = None,
        approximate_max_sample_count_to_obtain: Optional[int] = None,
    ):
        """Obtain and 'cure' a dataset to be used for fine-tuning of SafeLLaVA.

        Args:
            dataset (str): HuggingFace dataset ID
            destination_csv (str): Path to CSV file to offload datapoints
            media_key (Union[Callable, str]): _description_
            question_key (Union[Callable, str]): _description_
            answer_key (Union[Callable, str]): _description_
            default_media (Optional[str], optional): _description_. Defaults to None.
            default_question (Optional[str], optional): _description_. Defaults to None.
            default_answer (Optional[str], optional): _description_. Defaults to None.
            resume_enabled (bool, optional): _description_. Defaults to True.
            generate_samples_kwargs (Dict[str, Any], optional): _description_. Defaults to {}.
            max_rows_of_original_dataset_to_consider (Optional[int], optional): _description_. Defaults to None.
            approximate_max_sample_count_to_obtain (Optional[int], optional): _description_. Defaults to None.
        """

        # Load the HuggingFace dataset
        loaded_dataset = load_dataset(dataset)

        # Prepare variables
        num_samples_obtained = 0
        num_rows_already_processed = 0
        columns = list(VQADataPoint._fields) + ["original_dataset_index"]

        # Check if the destination_csv already exists
        if os.path.exists(destination_csv):
            # If resumption is disabled, we start fresh
            # by first deleting the previous file
            if not resume_enabled:
                os.remove(destination_csv)
            else:
                # Otherwise, make sure we pick up where we left off in this particular dataset.
                # We should make the assumption that each HuggingFace dataset will have its own
                # destination CSV file. So, we are assuming that the row of the sample we left
                # off on in the current CSV file is the row we should start at in the current
                # dataset.
                previous_df = pd.read_csv(destination_csv, sep='|')
                num_rows_already_processed = previous_df.iloc[-1]["original_dataset_index"] + 1
                num_samples_obtained = len(previous_df.index)
        
        # For all the samples in the HuggingFace dataset
        for idx, row in enumerate(loaded_dataset):
            # If we are past the maximum number of rows to consider
            # or have generated enough samples given the approximate
            # limit, break from iteration
            if max_rows_of_original_dataset_to_consider is not None and idx >= max_rows_of_original_dataset_to_consider or \
                approximate_max_sample_count_to_obtain is not None and num_samples_obtained >= approximate_max_sample_count_to_obtain:
                break

            # If we are resuming, continue at indices we have already obtained samples for
            if resume_enabled and idx < num_rows_already_processed:
                continue

            # Variables for this sampling round
            media = None
            question = None
            answer = None

            # Obtain the media
            if isinstance(media_key, str):
                media = row[media_key]
            elif media_key is not None:
                media = media_key(row)
            else:
                media = default_media
            
            # Obtain the question
            if isinstance(question_key, str):
                question = row[question_key]
            elif question_key is not None:
                question = question_key(row)
            else:
                question = default_question
            
            # Obtain the answer
            if isinstance(answer_key, str):
                answer = row[answer_key]
            elif answer_key is not None:
                answer = answer_key(row)
            else:
                answer = default_answer

            # Generate the Media-Text pair samples for this row
            samples = self.generate_samples_for_vqa_pair(
                media,
                question,
                answer,
                **generate_samples_kwargs
            )

            # Add to the number of samples we have obtained
            num_samples_obtained += len(samples)

            # Offload the samples to the destination CSV file
            pd.DataFrame(samples, columns=columns).to_csv(
                destination_csv,
                sep='|',
                mode='a',
            )

    def generate_samples_for_vqa_pair(
        self,
        media: str,
        question: str,
        answer: str,
        media_type: MediaType = MediaType.VIDEO,
        # Original text preservation args
        keep_original_vqa_pair: bool = True,
        use_vlm_to_determine_whether_original_vqa_is_safe: bool = True,
        # Non-private exposing description args
        create_description_without_private_attributes: bool = True,
        use_current_answer_as_description_response_but_rephrase_without_private_attributes: bool = False,
        description_templates: Set[str] = set(NON_PRIVATE_DESCRIPTION_TEMPLATES),
        # Refusal creation args
        create_refusals_for_private_attributes: bool = True,
        chance_to_create_refusal_per_attribute: float = 0.16667,
        private_attributes_to_protect: Set[str] = set(DEFAULT_PRIVATE_ATTRIBUTES_TO_PROTECT),
        refusal_question_templates: Set[str] = set(REFUSAL_QUESTION_TEMPLATES),
        refusal_answer_templates: Set[str] = set(REFUSAL_ANSWER_TEMPLATES),
        use_keywords_to_check_for_person: bool = True,
        keywords: Set[str] = set(STANDARD_KEYWORDS_FOR_PROMPTS_PERTAINING_TO_PEOPLE),
        use_vlm_to_check_for_person: bool = True,
        chance_for_vlm_to_rephrase_question_and_or_answer_from_template: float = 0.25,
        use_vlm_to_rephrase_question_and_or_answer_from_template: bool = False,
    ) -> List[VQADataPoint]:
        """Generate safe Media-Text (media + (question+answer)) pairs for the given media and/or its question and answer text.
        This method works by generating samples in three ways:

        1. Generating refusals for unsafe queries to the media.
        2. Generating a new, safe description according to a query from the templates, occasionally rephrased if enabled.
        3. Taking the original Media-Text pair and making it safe, if possible.

        Args:
            media (str): _description_
            question (str): _description_
            answer (str): _description_
            keep_original_vqa_pair (bool, optional): _description_. Defaults to True.
            use_vlm_to_determine_whether_original_vqa_is_safe (bool, optional): _description_. Defaults to True.
            create_description_without_private_attributes (bool, optional): _description_. Defaults to True.
            use_current_answer_as_description_response_but_rephrase_without_private_attributes (bool, optional): _description_. Defaults to False.
            description_templates (Set[str], optional): _description_. Defaults to set(NON_PRIVATE_DESCRIPTION_TEMPLATES).
            create_refusals_for_private_attributes (bool, optional): _description_. Defaults to True.
            chance_to_create_refusal_per_attribute (float, optional): _description_. Defaults to 0.16667.
            private_attributes_to_protect (Set[str], optional): _description_. Defaults to set(DEFAULT_PRIVATE_ATTRIBUTES_TO_PROTECT).
            refusal_question_templates (Set[str], optional): _description_. Defaults to set(REFUSAL_QUESTION_TEMPLATES).
            refusal_answer_templates (Set[str], optional): _description_. Defaults to set(REFUSAL_ANSWER_TEMPLATES).
            use_keywords_to_check_for_person (bool, optional): _description_. Defaults to True.
            keywords (Set[str], optional): _description_. Defaults to set(["person", "man", "woman", "boy", "girl", "baby"]).
            use_vlm_to_check_for_person (bool, optional): _description_. Defaults to True.
            chance_for_vlm_to_rephrase_question_and_or_answer_from_template (float, optional): _description_. Defaults to 0.25.
            use_vlm_to_rephrase_question_and_or_answer_from_template (bool, optional): _description_. Defaults to False.

        Returns:
            List[VQADataPoint]: A list of samples that were generated from the original Media-Text pair
        """

        # Media-Text pair samples
        media_text_pairs = []

        # Determine the media type for prompting
        media_category = media_type.value

        # Variable for discarding medias that do not contain people
        # and are thus not target samples of what we care to show
        # the MLLM for tuning
        contains_person = False

        # Check if there is a person in the media in a trivial manner
        if use_keywords_to_check_for_person and \
            any([(keyword in question or keyword in answer) for keyword in keywords]):
            contains_person = True

        # If we are still not sure if a person is in the media or not
        # and we run the VLM to give us a yes/no response if the setting
        # is enabled
        if not contains_person and use_vlm_to_check_for_person:
            contains_person = self.vlm.yes_or_no(media, f"Does the {media_category} contain one or more people?")

        # If the media does not contain a person, discard it.
        # There is certainly no privacy violation in this sample.
        if not contains_person:
            return []
        
        # If we are supposed to create refusals for this media
        if create_refusals_for_private_attributes:
            rephrase_question = False
            rephrase_answer = False

            # Check if we need to occasionally rephrase question and answer templates
            # to prevent overfitting on those templates
            if use_vlm_to_rephrase_question_and_or_answer_from_template:
                rephrase_question = (random.random() < chance_for_vlm_to_rephrase_question_and_or_answer_from_template)
                rephrase_answer = (random.random() < chance_for_vlm_to_rephrase_question_and_or_answer_from_template)

            # If we do not need to rephrase the question or answer templates
            if not rephrase_question and not rephrase_answer:
                # Simply append the datapoints as they are, with template
                # refusals for both question and answer
                media_text_pairs += list(filter(lambda x: x is not None, [
                    VQADataPoint(
                        media,
                        random.choice(refusal_question_templates).replace("{attribute}", attribute).replace("{media}", media_category),
                        random.choice(refusal_answer_templates).replace("{attribute}", attribute).replace("{media}", media_category),
                        AnswerType.REFUSAL,
                    ) if (random.random() < chance_to_create_refusal_per_attribute) else None \
                        for attribute in private_attributes_to_protect
                ]))
            else:
                # Pick a random question and answer refusal from the template
                original_question = random.choice(refusal_question_templates).replace("{media}", media_category)
                original_answer = random.choice(refusal_answer_templates).replace("{media}", media_category)

                # Rephrase the question simply to prevent overfitting on the templates
                rephrased_question = (self.vlm.rephrase(
                    media,
                    original_question,
                    extra_notes="Preserve the {attribute} tag in the same format in your rephrased sentence."
                ) if rephrase_question else original_question)
                rephrased_answer = (self.vlm.rephrase(
                    media,
                    original_answer,
                    extra_notes="Preserve the {attribute} tag in the same format in your rephrased sentence."
                ) if rephrase_answer else original_answer)

                # Append to the refusal Media-Text pair samples
                media_text_pairs += list(filter(lambda x: x is not None, [
                    VQADataPoint(
                        media,
                        rephrased_question.replace("{attribute}", attribute),
                        rephrased_answer.replace("{attribute}", attribute),
                        AnswerType.REFUSAL,
                    ) if (random.random() < chance_to_create_refusal_per_attribute) else None \
                        for attribute in private_attributes_to_protect
                ]))

        # If creating a custom description of the media is enabled
        if create_description_without_private_attributes:
            # Choose a random description question based on the templates
            description_question = random.choice(description_templates).replace("{media}", media_category)

            # If the answer given is the description we want to use
            if use_current_answer_as_description_response_but_rephrase_without_private_attributes:
                # Rephrase the answer in a safe manner
                description = self.vlm.rephrase(media, answer, extra_notes=PROTECTION_PROMPT.replace("{media}", media_category))
            else:
                # Generate a new, safe answer
                description = self.vlm(media, description_question).replace("{media}", media_category)

            # Append the Media-Text pair to the samples generated
            media_text_pairs.append(VQADataPoint(media, description_question, description, AnswerType.NORMAL))

        # If keeping the original VQA pair is allowed
        if keep_original_vqa_pair:

            # If the original VQA pair is potentially unsafe,
            # the following should be `True`
            if use_vlm_to_determine_whether_original_vqa_is_safe:
                
                # Make sure it is safe and does not include a question or
                # answer which contains exposure of private information
                vqa_is_unsafe = self.vlm.yes_or_no(
                    media,
                    "Do either of these statements contain a reference to any of the following attributes?\n\nAttributes:\n" + \
                        yaml.dumps(private_attributes_to_protect) + f"\n\nStatement 1: '{question}'\nStatement 2: '{answer}'"
                )

                if not vqa_is_unsafe:
                    # If the original VQA is safe, append the Media-Text pair to the samples as a normal response
                    media_text_pairs.append(VQADataPoint(media, question, answer, AnswerType.NORMAL))
                else:
                    # If the original VQA is unsafe, append the Media-Text pair to the samples as a refusal
                    media_text_pairs.append(VQADataPoint(media, question, "Sorry, but I cannot answer any question regarding personal information.", AnswerType.REFUSAL))
            else:
                # The original VQA pair is assumed to be safe, although we do not know if the original
                # VQA pair is a standard answer or a refusal
                media_text_pairs.append(VQADataPoint(media, question, answer, AnswerType.UNKNOWN))

        # Return all the Media-Text pair samples
        return media_text_pairs

def main():
    random.seed(87)

    datasets_to_curate = [
        {
            "dataset": "lmms-lab/ActivityNetQA",
            "destination_csv": "lmms-lab_ActivityNetQA",
            "media_key": lambda row: row["video_name"],
            "question_key": "question",
            "answer_key": "answer",
        },
        {
            "dataset": "lmms-lab/VideoDetailCaption",
            "destination_csv": "lmms-lab_VideoDetailCaption",
            "media_key": lambda row: row["video_name"],
            "question_key": "question",
            "answer_key": "answer",
        },
        {
            "dataset": "dai22dai/video",
            "destination_csv": "dai22dai_video",
            "media_key": lambda row: row["image"],
            "default_question": None,
            "default_answer": None,
            "generate_samples_kwargs": {
                "media_type": MediaType.IMAGE,
                "use_vlm_to_check_for_person": False,
                "keep_original_vqa_pair": False,
            },
        },
        # {
        #     "dataset": "LanguageBind/Open-Sora-Plan-v1.2.0",
        #     "destination_csv": "LanguageBind_Open-Sora-Plan-v1.2.0",
        #     "answer_key": "cap",
        #     "generate_samples_kwargs": {
        #         "use_vlm_to_check_for_person": False,
        #         "use_current_answer_as_description_response_but_rephrase_without_private_attributes": True,
        #     },
        # },
    ]

    vlm = Phi_3_5_Multimodal()
    curator = DataCuratorConstruct(vlm)

    for dataset_kwargs in datasets_to_curate:
        curator.curate_dataset(**dataset_kwargs)

if __name__ == "__main__":
    main()
