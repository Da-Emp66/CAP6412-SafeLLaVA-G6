import argparse
import os
import random
import re
from typing import List, Literal, Optional, Set
from datasets import load_dataset
import kagglehub
import tarfile
import pandas as pd
import yaml
from safellava.dataset.dataset_fabrication import AnswerType, VQADataCuratorConstruct, VQADataPoint
from safellava.interfaces import BaseMultiModalLanguageModel
from safellava.models import QwenVL_Instruct
from safellava.models.models import instantiate_model_based_on_model_map
from safellava.utils import MediaType, load_online_files

#####################################################
# Privacy Globals
#####################################################

# Protection
PROTECTION_PROMPT = """DO NOT describe the person or people in the {media} by anything other than `person`, `people`, or that person's or persons' occupation(s). Keep the overall {media} description detailed while excluding information about anyone in the {media}'s name, age, sex, gender, pronouns, race, hair color, hairstyle, facial or other physical characteristics."""

# Attributes
DEFAULT_PRIVATE_ATTRIBUTES_TO_PROTECT = [
    "name",
    "age",
    "race",
    "sex",
    "gender",
    "pronoun(s)",
    "hair color",
    "hairstyle",
    "facial or other physical characteristics",
    "uniquely identifiable information",
]

# Keywords

NEUTRAL_KEYWORDS_PERTAINING_TO_PEOPLE = [
    "person",
    "individual",
]

NEUTRAL_KEYWORDS_PERTAINING_TO_PEOPLE_PLURAL = [
    "people",
    "individuals",
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

STANDARD_KEYWORDS_FOR_PROMPTS_PERTAINING_TO_PEOPLE_PLURAL = [
    "people",
    "men",
    "males",
    "women",
    "females",
    "boys",
    "girls",
    "adults",
    "children",
    "babies",
]

# Pronouns
PROTECTED_PRONOUNS = {
    "noun1": [
        "he",
        "she",
    ],
    "noun2": [
        "him",
        "her",
    ],
    "possessive": [
        "his",
        "her",
    ],
}

REPLACEMENT_NEUTRAL_PRONOUNS = {
    "noun1": [
        "he or she",
        "the person",
    ],
    "noun2": [
        "him or her",
        "the person",
    ],
    "possessive": [
        "his or her",
        "the person's",
    ],
}

REPLACEMENT_NEUTRAL_PRONOUNS_WITH_GRAMMATICAL_LENIENCY = {
    "noun1": [
        "he or she",
        "they",
        "the person",
    ],
    "noun2": [
        "him or her",
        "them",
        "the person",
    ],
    "possessive": [
        "his or her",
        "theirs",
        "the person's",
    ],
}

# Refusals
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

# Descriptions
NON_PRIVATE_DESCRIPTION_TEMPLATES = [
    f"Describe the {{media}}. {PROTECTION_PROMPT}",
]

# Cleaning
def classical_remove_private_attributes_from_sentence(
    original_sentence: str,
    grammatically_lenient_replacement_pronouns: bool = True
) -> str:
    processed_sentence = re.sub(fr"({'|'.join(STANDARD_KEYWORDS_FOR_PROMPTS_PERTAINING_TO_PEOPLE_PLURAL)})", random.choice(NEUTRAL_KEYWORDS_PERTAINING_TO_PEOPLE_PLURAL), original_sentence)
    processed_sentence = re.sub(fr"({'|'.join(STANDARD_KEYWORDS_FOR_PROMPTS_PERTAINING_TO_PEOPLE)})", random.choice(NEUTRAL_KEYWORDS_PERTAINING_TO_PEOPLE), original_sentence)
    pronoun_replacement_options = (REPLACEMENT_NEUTRAL_PRONOUNS_WITH_GRAMMATICAL_LENIENCY if grammatically_lenient_replacement_pronouns else REPLACEMENT_NEUTRAL_PRONOUNS)
    for pronoun_type in PROTECTED_PRONOUNS:
        processed_sentence = re.sub(fr"({'|'.join(PROTECTED_PRONOUNS[pronoun_type])})", random.choice(pronoun_replacement_options[pronoun_type]), processed_sentence)
    return processed_sentence

#####################################################
# Non-HuggingFace Dataset Loaders
#####################################################

def load_visogender(
    _dataset_name: str = "visogender",
    urls: List[str] = [
        "https://raw.githubusercontent.com/oxai/visogender/refs/heads/main/data/visogender_data/OO/OO_Visogender_02102023.tsv",
        "https://raw.githubusercontent.com/oxai/visogender/refs/heads/main/data/visogender_data/OP/OP_Visogender_02102023.tsv",
        "https://raw.githubusercontent.com/oxai/visogender/refs/heads/main/data/visogender_data/OP/OP_Visogender_11012024.tsv",
    ],
):
    files = load_online_files(urls)
    return load_dataset("tsv", data_files=files)

def load_hollywood2(
    dataset_name: str = "hollywood2",
    urls: List[str] = [
        "ftp://ftp.irisa.fr/local/vistas/actions/Hollywood2-actions.tar.gz",
        "ftp://ftp.irisa.fr/local/vistas/actions/Hollywood2-scenes.tar.gz",
    ],
    download_dir: Optional[str] = None,
    filename_filter: Optional[List[str]] = None,
):
    # Ensure initial variables and directories are prepared
    if download_dir is None:
        download_dir = os.path.join("data_downloads", dataset_name)

    os.makedirs(download_dir, exist_ok=True)

    output_csv = os.path.join(download_dir, f"{dataset_name}.csv")
    videos = []

    output_csv_exists = os.path.join(output_csv)

    # Download and extract the files
    files = load_online_files(urls, downloads_dir=download_dir)
    for file in files:
        unzipped_output_dir = file.rstrip(".tar.gz")
        opened_tar = tarfile.open(file)
        if not os.path.exists(unzipped_output_dir):
            opened_tar.extractall(unzipped_output_dir)

        if not output_csv_exists:
            # Gather video paths to make the dataset
            videos_dir = os.path.join(unzipped_output_dir, "Hollywood2", "AVIClips")
            current_videos = os.listdir(videos_dir)
            current_videos = [os.path.join(videos_dir, video) for video in current_videos]
            videos += current_videos

    if not output_csv_exists:
        # Filter the necessary videos
        if filename_filter is not None:
            videos = list(
                filter(
                    lambda video_filename: (
                        not any([(filtered_string in video_filename) for filtered_string in filename_filter])
                    ),
                    videos
                )
            )

        # Write the dataset CSV
        pd.DataFrame({ "video": videos }).to_csv(output_csv)

    # Return the dataset loader
    return load_dataset("csv", data_files=[output_csv])

def load_video_story(
    _dataset_name: str = "video_story",
    urls: List[str] = [
        "https://isis-data.science.uva.nl/mediamill/videostory/vs_v1.tar.gx",
    ],
):
    files = load_online_files(urls)
    for file in files:
        output_dir = file.rstrip(".tar.gx")
        opened_tar = tarfile.open(file)
        if not os.path.exists(output_dir):
            opened_tar.extractall(output_dir)
    
    raise NotImplementedError()

def load_vatex_video_captioning(
    _dataset_name: str = "vatex",
    urls: List[str] = [
        "https://eric-xw.github.io/vatex-website/data/vatex_training_v1.0.json",
    ],
):
    files = load_online_files(urls)
    raise NotImplementedError()
    
def load_youtube_pose(
    _dataset_name: str = "youtube-pose",
    kaggle_handle: str = "soumikrakshit/youtube-pose-dataset",
):
    path = kagglehub.dataset_download(kaggle_handle)
    print("Path to dataset files:", path)
    raise NotImplementedError()

def load_condensed_movies(
    _dataset_name: str = "condensed_movies",
    urls: List[str] = [
        "https://raw.githubusercontent.com/m-bain/CondensedMovies/refs/heads/master/data/metadata/clips.csv",
        "https://raw.githubusercontent.com/m-bain/CondensedMovies/refs/heads/master/data/metadata/descriptions.csv",
        "https://raw.githubusercontent.com/m-bain/CondensedMovies/refs/heads/master/data/metadata/durations.csv",
        "https://github.com/m-bain/CondensedMovies/blob/master/data/metadata/movie_info.csv",
    ],
):
    files = load_online_files(urls)
    raise NotImplementedError()

def load_narrated_instruction_videos(
    _dataset_name: str = "narrated_instruction_videos",
    urls: List[str] = [
        "https://www.di.ens.fr/willow/research/instructionvideos/data_new.tar.gz",
    ],
):
    files = load_online_files(urls)
    raise NotImplementedError()

def load_coin_dataset(
    _dataset_name: str = "coin_dataset",
    urls: List[str] = [
        "https://raw.githubusercontent.com/coin-dataset/annotations/refs/heads/master/COIN.json",
    ],
):
    # https://coin-dataset.github.io/
    raise NotImplementedError()

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
    description_templates: List[str] = list(NON_PRIVATE_DESCRIPTION_TEMPLATES),
    classically_clean_description: bool = True,
    # Refusal creation args
    create_refusals_for_private_attributes: bool = True,
    chance_to_create_refusal_per_attribute: float = 0.16667,
    private_attributes_to_protect: List[str] = list(DEFAULT_PRIVATE_ATTRIBUTES_TO_PROTECT),
    refusal_question_templates: List[str] = list(REFUSAL_QUESTION_TEMPLATES),
    refusal_answer_templates: List[str] = list(REFUSAL_ANSWER_TEMPLATES),
    must_contain_person: bool = True,
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
        description_templates (List[str], optional): _description_. Defaults to list(NON_PRIVATE_DESCRIPTION_TEMPLATES).
        classically_clean_description (bool, optional): _description_. Defaults to True.
        create_refusals_for_private_attributes (bool, optional): _description_. Defaults to True.
        chance_to_create_refusal_per_attribute (float, optional): _description_. Defaults to 0.16667.
        private_attributes_to_protect (List[str], optional): _description_. Defaults to list(DEFAULT_PRIVATE_ATTRIBUTES_TO_PROTECT).
        refusal_question_templates (List[str], optional): _description_. Defaults to list(REFUSAL_QUESTION_TEMPLATES).
        refusal_answer_templates (List[str], optional): _description_. Defaults to list(REFUSAL_ANSWER_TEMPLATES).
        must_contain_person (bool, optional): _description_. Defaults to True.
        use_keywords_to_check_for_person (bool, optional): _description_. Defaults to True.
        keywords (Set[str], optional): _description_. Defaults to set(STANDARD_KEYWORDS_FOR_PROMPTS_PERTAINING_TO_PEOPLE).
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

    if must_contain_person:
        # Variable for discarding medias that do not contain people
        # and are thus not target samples of what we care to show
        # the MLLM for tuning
        contains_person = False

        # Check if there is a person in the media in a trivial manner
        if use_keywords_to_check_for_person and \
            any([((question is not None and keyword in question) or (answer is not None and keyword in answer)) for keyword in keywords]):
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
        
        # Ensure for a fact that the description is safe
        if classically_clean_description:
            description = classical_remove_private_attributes_from_sentence(description)

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

def process_dataset(
    dataset_names: str,
    process: Literal["curate", "clean"],
    model: str,
):
    dataset_names = dataset_names.split(",")
    random.seed(87)

    if process == "curate":
        curatable_datasets = [
            
            #####################################################
            # Videos
            #####################################################

            {
                "dataset": "hollywood2",
                "media_key": "video",
                "dataset_obtain_strategy": load_hollywood2,
                "dataset_obtain_kwargs": {
                    "urls": [
                        "ftp://ftp.irisa.fr/local/vistas/actions/Hollywood2-actions.tar.gz"
                    ],
                    "filename_filter": [
                        "autoauto"
                    ],
                },
                "generate_samples_kwargs": {
                    "must_contain_person": False,
                },
            },
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
            
            #####################################################
            # Images
            #####################################################

            {
                "dataset": "dai22dai/video",
                "media_key": lambda row: row["image"],
                "approximate_max_sample_count_to_obtain": 200,
                "generate_samples_kwargs": {
                    "media_type": MediaType.IMAGE,
                    "must_contain_person": False,
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
                    "must_contain_person": False,
                }
            },
        ]

        vlm = instantiate_model_based_on_model_map(model)

        print(f"Using {model} to curate dataset...")

        curator = VQADataCuratorConstruct(vlm)

        for dataset_kwargs in curatable_datasets:
            if dataset_kwargs["dataset"] not in dataset_names:
                continue

            curator.curate_dataset(
                generate_samples_strategy=generate_samples_for_vqa_pair,
                **dataset_kwargs,
            )

    else:

        curator = VQADataCuratorConstruct(vlm)

        for dataset in dataset_names:
            curator


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--datasets",
        type=str,
        help="The names of the datasets; comma delimited",
        required=True,
    )
    parser.add_argument(
        "-p",
        "--process",
        type=str,
        help="Process to perform on dataset",
        choices=[
            "curate",
            "clean",
        ],
        required=True,
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        help="Model to try out",
        choices=[
            "Qwen2-VL",
            "Qwen2.5-VL",
            "Phi-3.5-Multimodal",
            "Ovis2-1B",
            "Ovis2-2B",
            "Ovis2-4B",
            "MiniCPM-o-2_6",
            "Llava-OneVision-Qwen2-0.5B",
            "Llava-Interleave-Qwen2-0.5B",
        ],
        default="Ovis2-1B",
        required=False,
    )
    args = parser.parse_args()
    
    process_dataset(
        args.dataset,
        args.process,
        args.model,
    )
