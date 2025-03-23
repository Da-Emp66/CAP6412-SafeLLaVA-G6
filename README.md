# CAP6412-SafeLLaVA-G6
A research effort into removing Personally Identifiable Information (PII) from Multimodal Large Language Model (MLLM) outputs.


### Install

#### Package

    # Using pip
    pip install git+https://github.com/Da-Emp66/CAP6412-SafeLLaVA-G6.git

#### From Source

    # Via script
    . install-local.sh

    # Or direct
    curl -LsSf https://astral.sh/uv/install.sh | sh # or pip install uv
    cd src/safellava && uv sync


### Folder Structure

The folder structure of this project is organized as follows:

    ./
     |---notebooks/      # Colab notebooks and recipes
     |---src/safellava/     # Python package source code folder, importable in notebooks


### Use

#### How to Use Your Own Custom Dataset
Modify `src/safellava/datasets/privacy_dataset.py` with your own dataset parameters in the `curatable_datasets` list and, optionally, your own dataset loader for downloading and loading your dataset, if it cannot be completely downloaded, including all videos and/or images, from HuggingFace.

```python
from datasets import load_dataset

...

# If you want to use a dataset that is not a dataset that can be
# downloaded directly and completely from HuggingFace
def load_my_dataset(dataset: str, **dataset_obtain_kwargs):
    # Download files from my dataset
    ... # Can use functions from `utils.py` like load_online_files(urls=[...]) if all URLs are known
    # or download_youtube_video(video_id="YouTube ID for video") if downloading YouTube videos,
    # or you can write your own functions for downloading files for your dataset

    # Return the dataset as a loaded
    return load_dataset("csv", data_files=["my_custom_dataset.csv"])

...

def main(args):

    ...

    curatable_datasets = [

        ...,

        {
            "dataset": "my_dataset", # Choose a dataset name, or use a Huggingface dataset
            "dataset_obtain_strategy": load_my_dataset, # If you chose a HuggingFace dataset for "dataset"
            # you do not have to implement a custom loader, but if your dataset is from a different source,
            # you do have to implement a dataset loader.
            "dataset_obtain_kwargs": { ... }, # These are the kwargs that get passed to your `dataset_obtain_strategy`. They will be passed like
            # dataset_obtain_strategy(dataset, **dataset_obtain_kwargs)
            "generate_samples_kwargs": { ... }, # This will be passed to `generate_samples_for_vqa_pair` like
            # generate_samples_strategy(vlm, media, question, answer, **generate_samples_kwargs)
            "media_key": "video", # This tells the curator how to access the video or image from a row of
            # your dataset. It can be a callable like `lambda row: row["image"]` or a string or None if
            # you do not want media to be passed during inference.
            "question_key": "question", # Similar to the above, but for the question in your dataset
            "answer_key": "answer" # Similar to the above, but for the answer in your dataset
        },

        ...,

    ]

    ...

```
Once you configure all the parameters you wish for dataset curation and optionally make your custom loader function, the curator construct will generate samples for you using combinations of MLLM-generated privacy-focused descriptions and random refusals for requests to expose private information in videos.

One of the processes for our dataset curation and tuning is as follows:

    # Navigate to the proper directory
    cd src/safellava

    # Write your custom dataset curator parameters and dataset loader, as above

    # Curate a dataset - This is the final step in creating your own scrubbed dataset!
    uv run dataset/privacy_dataset.py -p curate -d hollywood2

    ### The following steps are advanced steps involved in tuning on your own generated dataset.

    # Merge two curated datasets
    uv run dataset/privacy_dataset.py -p merge -d privacy_preservation/activitynet_curated/datapoints.csv,privacy_preservation/hollywood2_curated/datapoints.csv

    # Clean a dataset
    uv run dataset/privacy_dataset.py -p clean -d activitynet_curated_datapoints-hollywood2_curated_datapoints_merged.csv

    # Prepare a dataset for tuning
    uv run dataset/privacy_dataset.py -p clean -d activitynet_curated_datapoints-hollywood2_curated_datapoints_merged_cleaned.csv

    # Upload the dataset to your Google Drive

    # Go to and run the tuning notebook https://colab.research.google.com/drive/1r_1e4Opo6GB1QeOvgoCs0FRQljE_gyRc?usp=sharing
    # This should give you a model as output.

    # Infer on your tuned model
    uv run dataset/privacy_dataset.py -m tuned -t <path-to-tuned-model-checkpoint>

