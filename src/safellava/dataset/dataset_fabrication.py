from enum import Enum
import json
import os
import shutil
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Union
import warnings
from datasets import load_dataset

import pandas as pd

from safellava.interfaces import BaseMultiModalLanguageModel
from safellava.utils import MediaType, get_media_type

class AnswerType(Enum):
    NORMAL = 0
    REFUSAL = 1
    UNKNOWN = 2

class VQADataPoint(NamedTuple):
    media_path: str
    question: str
    answer: str
    answer_type: AnswerType

class VQADataCuratorConstruct:
    def __init__(self, vlm: Optional[BaseMultiModalLanguageModel] = None):
        self.vlm = vlm

    def curate_dataset(
        self,
        dataset: str,
        dataset_obtain_strategy: Callable = load_dataset,
        dataset_obtain_kwargs: Dict[str, Any] = {},
        destination_directory: Optional[str] = None,
        destination_csv: Optional[str] = os.path.join("{destination_directory}", "datapoints.csv"),
        images_directory: Optional[str] = os.path.join("{destination_directory}", "images"),
        videos_directory: Optional[str] = os.path.join("{destination_directory}", "videos"),
        unknown_media_directory: Optional[str] = os.path.join("{destination_directory}", "unknown_media"),
        media_key: Optional[Union[Callable, str]] = None,
        question_key: Optional[Union[Callable, str]] = None,
        answer_key: Optional[Union[Callable, str]] = None,
        default_media: Optional[str] = None,
        default_question: Optional[str] = None,
        default_answer: Optional[str] = None,
        resume_enabled: bool = True,
        generate_samples_strategy: Callable = None,
        generate_samples_kwargs: Dict[str, Any] = {},
        max_rows_of_original_dataset_to_consider: Optional[int] = None,
        approximate_max_sample_count_to_obtain: Optional[int] = None,
    ):
        """Obtain and 'cure' a dataset to be used for fine-tuning of SafeLLaVA.

        Args:
            dataset (str): HuggingFace dataset ID or dataset name
            dataset_obtain_strategy (Callable, optional): _description_. Defaults to load_dataset.
            dataset_obtain_kwargs (Dict[str, Any], optional): _description_. Defaults to {}.
            destination_directory (Optional[str], optional): Path to directory to store curated data. Defaults to f"./{dataset}_curated".replace('/', "_").
            destination_csv (Optional[str], optional): Path to CSV file to offload datapoints. Defaults to os.path.join("{destination_directory}", "datapoints.csv").
            images_directory (Optional[str], optional): Path to directory to store curated images. Defaults to os.path.join("{destination_directory}", "images").
            videos_directory (Optional[str], optional): Path to directory to store curated videos. Defaults to os.path.join("{destination_directory}", "videos").
            unknown_media_directory (Optional[str], optional): Path to directory to store curated media that was not identified as an image or video. Defaults to os.path.join("{destination_directory}", "unknown_media").
            media_key (Optional[Union[Callable, str]], optional): _description_
            question_key (Optional[Union[Callable, str]], optional): _description_
            answer_key (Optional[Union[Callable, str]], optional): _description_
            default_media (Optional[str], optional): _description_. Defaults to None.
            default_question (Optional[str], optional): _description_. Defaults to None.
            default_answer (Optional[str], optional): _description_. Defaults to None.
            resume_enabled (bool, optional): _description_. Defaults to True.
            generate_samples_strategy (Callable, optional): _description_. Defaults to self.generate_samples_for_vqa_pair.
            generate_samples_kwargs (Dict[str, Any], optional): _description_. Defaults to {}.
            max_rows_of_original_dataset_to_consider (Optional[int], optional): _description_. Defaults to None.
            approximate_max_sample_count_to_obtain (Optional[int], optional): _description_. Defaults to None.
        """
        
        # Identify the default directory to place results
        if destination_directory is None:
            destination_directory = "./" + (f"{dataset}_curated".replace('/', "_"))

        # Formalize paths to curation results
        destination_csv = destination_csv.replace("{destination_directory}", destination_directory.rstrip('/'))
        images_directory = images_directory.replace("{destination_directory}", destination_directory.rstrip('/'))
        videos_directory = videos_directory.replace("{destination_directory}", destination_directory.rstrip('/'))
        unknown_media_directory = unknown_media_directory.replace("{destination_directory}", destination_directory.rstrip('/'))

        # Ensure the method for generating samples has been set
        if generate_samples_strategy is None:
            raise ValueError("Argument `generate_samples_strategy` for `curate_dataset` must not be `None`.")

        # Obtain the HuggingFace or other dataset
        loaded_dataset = dataset_obtain_strategy(
            dataset,
            **dataset_obtain_kwargs,
        )["train"]

        # Prepare variables
        num_samples_obtained = 0
        num_rows_already_processed = 0
        columns = list(VQADataPoint._fields) + ["original_dataset_index"]
        current_df = pd.DataFrame()

        # Check if the destination_csv already exists
        if os.path.exists(destination_csv):
            # If resumption is disabled, we start fresh
            # by first deleting the previous file
            if not resume_enabled:
                os.remove(destination_csv)
            else:
                # Otherwise, make sure we pick up where we left off in this particular dataset.
                # We should make the assumption that each HuggingFace or other dataset will have its own
                # destination CSV file. So, we are assuming that the row of the sample we left
                # off on in the current CSV file is the row we should start at in the current
                # dataset.
                previous_df = pd.read_csv(destination_csv, sep='|')
                num_rows_already_processed = int(previous_df.iloc[-1]["original_dataset_index"]) + 1
                num_samples_obtained = len(previous_df.index)
                current_df = previous_df
                print(f"Found current dataframe at `{destination_csv}`. Resuming at dataset index `{num_rows_already_processed}`.")

        # For all the samples in the HuggingFace or other dataset
        for idx, row in enumerate(loaded_dataset):
            try:
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
                samples = generate_samples_strategy(
                    self.vlm,
                    media,
                    question,
                    answer,
                    **generate_samples_kwargs,
                )

                # Copy the media into the proper directory
                media_type = get_media_type(media)

                if media_type == MediaType.IMAGE:
                    os.makedirs(images_directory, exist_ok=True)
                    media = str(shutil.copy(media, images_directory))
                elif media_type == MediaType.VIDEO:
                    os.makedirs(videos_directory, exist_ok=True)
                    media = str(shutil.copy(media, videos_directory))
                else:
                    warnings.warn(f"Media type of {media} unknown.")
                    os.makedirs(unknown_media_directory, exist_ok=True)
                    media = str(shutil.copy(media, unknown_media_directory))

                # Fix the media path in the newly generated samples
                for idx, sample in enumerate(samples):
                    sample.media_path = media
                    samples[idx] = sample

                # Convert the new samples into a format readable by pandas (listify the named tuples)
                samples = [(list(sample[:-1]) + [sample[-1].value] + [idx]) for sample in samples]

                # Add to the number of samples we have obtained
                num_samples_obtained += len(samples)

                # Store the new samples
                current_df = pd.concat([current_df, pd.DataFrame(samples, columns=columns)], ignore_index=True)

                # Offload the samples to the destination CSV file
                current_df.to_csv(
                    destination_csv,
                    sep='|',
                )
            except Exception as e:
                print(f"Error `{e.__class__}` at dataset index `{idx}`. Skipping...")
    
    def load_existing_dataset(
        self,
        dataset_csv: str,
        drop_columns: List[str],
    ):
        loaded_dataset = load_dataset("csv", data_files=[dataset_csv], delimiter="|")["train"]
        loaded_dataset = loaded_dataset.remove_columns(drop_columns)
        return loaded_dataset
    
    def write_csv_of_loaded_existing_dataset(
        self,
        loaded_dataset: Any,
        destination_csv: str,
    ):
        loaded_dataset = loaded_dataset.remove_columns("Unnamed: 0")

        loaded_dataset.to_pandas().to_csv(
            destination_csv,
            sep='|',
            index=True,
        )

    def postprocess_existing_datasets(
        self,
        dataset_csv: List[str],
        destination_csv: List[str],
        postprocess_media: Optional[Callable] = lambda x: x,
        postprocess_question: Optional[Callable] = lambda x: x,
        postprocess_answer: Optional[Callable] = lambda x: x,
        postprocess_answer_type: Optional[Callable] = lambda x: x,
    ):
        postprocessing_funcs_to_keys = [
            (postprocess_media, "media_path"),
            (postprocess_question, "question"),
            (postprocess_answer, "answer"),
            (postprocess_answer_type, "answer_type"),
        ]

        loaded_dataset = load_dataset("csv", data_files=[dataset_csv], delimiter="|")["train"]
        
        def alter_row(row):
            for postprocessing_func, key in postprocessing_funcs_to_keys:
                row[key] = postprocessing_func(row[key])
            return row

        loaded_dataset = loaded_dataset.map(alter_row)

        self.write_csv_of_loaded_existing_dataset(
            loaded_dataset,
            destination_csv,
        )

    def merge_existing_datasets(
        self,
        *dataset_csvs: List[str],
        destination_csv: str
    ) -> str:
        loaded_dataset = load_dataset("csv", data_files=dataset_csvs, delimiter="|")["train"]
        
        self.write_csv_of_loaded_existing_dataset(
            loaded_dataset,
            destination_csv,
        )

    def prepare_dataset_for_swift_tuning(
            self,
            dataset_csv: str,
            destination_json: str,
        ):
        """Convert each row into a line like:
        {"messages": [{"role": "system", "content": "You are a helpful and harmless assistant."}, {"role": "user", "content": "<image>What is in the image, <video>What is in the video?"}, {"role": "assistant", "content": "The image shows an elephant, and the video shows a puppy running on the grass."}], "images": ["/xxx/x.jpg"], "videos": ["/xxx/x.mp4"]}
        and place the rows into an improper JSON file
        
        Args:
            dataset_csv (str): _description_
        """
        
        loaded_dataset = load_dataset("csv", data_files=[dataset_csv], delimiter='|')["train"]

        lines = []
        for idx, row in enumerate(loaded_dataset):
            media_type = get_media_type(row["media_path"])

            if media_type == MediaType.IMAGE_OR_VIDEO:
                continue

            lines.append(json.dumps({
                "messages": [
                    {"role": "system", "content": "You are a helpful and harmless assistant."},
                    {"role": "user", "content": f"<{media_type.value}>{row['question']}"},
                    {"role": "assistant", "content": row["answer"]},
                ],
                "videos": [row["media_path"]] if media_type == MediaType.VIDEO else [],
                "images": [row["media_path"]] if media_type == MediaType.IMAGE else [],
            }))

        lines = list(map(lambda json_datapoint: f"{json_datapoint}\n", lines))

        with open(destination_json, "w") as dest:
            dest.writelines(lines)
            dest.close()
        