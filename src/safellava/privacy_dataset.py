import random
from typing import List, Optional, Set
from datasets import load_dataset
import yaml
from safellava.dataset_fabrication import AnswerType, DataCuratorConstruct, VQADataPoint
from safellava.interfaces import BaseMultiModalLanguageModel
from safellava.models import Phi_3_5_Multimodal
from safellava.utils import MediaType, load_online_files

#####################################################
# Privacy Globals
#####################################################

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
    "male",
    "woman",
    "female",
    "boy",
    "girl",
    "adult",
    "child",
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

#####################################################
# Non-HuggingFace Dataset Loaders
#####################################################

def load_visogender(
        _dataset_name: str,
        urls: List[str] = [
            "https://raw.githubusercontent.com/oxai/visogender/refs/heads/main/data/visogender_data/OO/OO_Visogender_02102023.tsv",
            "https://raw.githubusercontent.com/oxai/visogender/refs/heads/main/data/visogender_data/OP/OP_Visogender_02102023.tsv",
            "https://raw.githubusercontent.com/oxai/visogender/refs/heads/main/data/visogender_data/OP/OP_Visogender_11012024.tsv",
        ],
    ):

    files = load_online_files(urls)

    return load_dataset("csv", data_files=files)

#####################################################
# Sample Generation
#####################################################

def generate_samples_for_vqa_pair(
    vlm: BaseMultiModalLanguageModel,
    media: str,
    question: Optional[str] = None,
    answer: Optional[str] = None,
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
        vlm (BaseMultiModalLanguageModel): _description_
        media (str): _description_
        question (str, optional): _description_
        answer (str, optional): _description_
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
        contains_person = vlm.yes_or_no(media, f"Does the {media_category} contain one or more people?")

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
            rephrased_question = (vlm.rephrase(
                media,
                original_question,
                extra_notes="Preserve the {attribute} tag in the same format in your rephrased sentence."
            ) if rephrase_question else original_question)
            rephrased_answer = (vlm.rephrase(
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
        if answer is not None and use_current_answer_as_description_response_but_rephrase_without_private_attributes:
            # Rephrase the answer in a safe manner
            description = vlm.rephrase(media, answer, extra_notes=PROTECTION_PROMPT.replace("{media}", media_category))
        else:
            # Generate a new, safe answer
            description = vlm(media, description_question).replace("{media}", media_category)

        # Append the Media-Text pair to the samples generated
        media_text_pairs.append(VQADataPoint(media, description_question, description, AnswerType.NORMAL))

    # If keeping the original VQA pair is allowed
    if question is not None and answer is not None and keep_original_vqa_pair:

        # If the original VQA pair is potentially unsafe,
        # the following should be `True`
        if use_vlm_to_determine_whether_original_vqa_is_safe:
            
            # Make sure it is safe and does not include a question or
            # answer which contains exposure of private information
            vqa_is_unsafe = vlm.yes_or_no(
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

#####################################################
# Run Dataset Curator
#####################################################

def main():
    random.seed(87)

    datasets_to_curate = [
        {
            "dataset": "lmms-lab/ActivityNetQA",
            "media_key": lambda row: row["video_name"],
            "question_key": "question",
            "answer_key": "answer",
            "approximate_max_sample_count_to_obtain": 200,
        },
        {
            "dataset": "lmms-lab/VideoDetailCaption",
            "media_key": lambda row: row["video_name"],
            "question_key": "question",
            "answer_key": "answer",
            "approximate_max_sample_count_to_obtain": 200,
        },
        {
            "dataset": "dai22dai/video",
            "media_key": lambda row: row["image"],
            "approximate_max_sample_count_to_obtain": 200,
            "generate_samples_kwargs": {
                "media_type": MediaType.IMAGE,
                "use_vlm_to_check_for_person": False,
                "keep_original_vqa_pair": False,
            },
        },
        {
            "dataset": "visogender",
            "dataset_obtain_strategy": load_visogender,
            "dataset_obtain_kwargs": {
                "urls": [
                    "https://raw.githubusercontent.com/oxai/visogender/refs/heads/main/data/visogender_data/OO/OO_Visogender_02102023.tsv",
                    "https://raw.githubusercontent.com/oxai/visogender/refs/heads/main/data/visogender_data/OP/OP_Visogender_02102023.tsv",
                    "https://raw.githubusercontent.com/oxai/visogender/refs/heads/main/data/visogender_data/OP/OP_Visogender_11012024.tsv",
                ],
            },
            "media_key": "URL type (Type NA if can't find)",
            "approximate_max_sample_count_to_obtain": 200,
            "generate_samples_kwargs": {
                "media_type": MediaType.IMAGE,
                "use_vlm_to_check_for_person": False,
            }
        },
        {

        },
    ]

    vlm = Phi_3_5_Multimodal()
    curator = DataCuratorConstruct(vlm)

    for dataset_kwargs in datasets_to_curate:
        curator.curate_dataset(
            generate_samples_strategy=generate_samples_for_vqa_pair,
            **dataset_kwargs)

if __name__ == "__main__":
    main()
