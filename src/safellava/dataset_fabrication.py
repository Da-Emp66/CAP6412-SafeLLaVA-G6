from enum import Enum
import os
from typing import Any, Callable, Dict, NamedTuple, Optional, Union
from datasets import load_dataset

import pandas as pd

from safellava.interfaces import BaseMultiModalLanguageModel

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
        dataset_obtain_strategy: Callable = load_dataset,
        dataset_obtain_kwargs: Dict[str, Any] = {},
        destination_csv: Optional[str] = None,
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
            destination_csv (Optional[str], optional): Path to CSV file to offload datapoints. Defaults to f"{dataset}_curated.csv".replace('/', "_").
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

        if destination_csv is None:
            destination_csv = f"{dataset}_curated.csv".replace('/', "_")

        if generate_samples_strategy is None:
            raise ValueError("Argument `generate_samples_strategy` for `curate_dataset` must not be `None`.")

        # Obtain the HuggingFace or other dataset
        loaded_dataset = dataset_obtain_strategy(
            dataset,
            **dataset_obtain_kwargs
        )

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
                # We should make the assumption that each HuggingFace or other dataset will have its own
                # destination CSV file. So, we are assuming that the row of the sample we left
                # off on in the current CSV file is the row we should start at in the current
                # dataset.
                previous_df = pd.read_csv(destination_csv, sep='|')
                num_rows_already_processed = previous_df.iloc[-1]["original_dataset_index"] + 1
                num_samples_obtained = len(previous_df.index)
        
        # For all the samples in the HuggingFace or other dataset
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
            samples = generate_samples_strategy(
                self.vlm,
                media,
                question,
                answer,
                **generate_samples_kwargs,
            )

            # Add to the number of samples we have obtained
            num_samples_obtained += len(samples)

            # Offload the samples to the destination CSV file
            pd.DataFrame(samples, columns=columns).to_csv(
                destination_csv,
                sep='|',
                mode='a',
            )
