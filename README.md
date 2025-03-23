# CAP6412-SafeLLaVA-G6
A research effort into removing Personally Identifiable Information (PII) from Multimodal Large Language Model (MLLM) outputs.


### Install

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

One of the processes for our dataset curation and tuning is as follows:

    # Navigate to the proper directory
    cd src/safellava

    # Curate a dataset
    uv run dataset/privacy_dataset.py -p curate -d hollywood2

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

