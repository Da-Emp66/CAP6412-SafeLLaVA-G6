#!/bin/bash

init_conda () {
    ~/miniconda3/bin/conda init bash
    ~/miniconda3/bin/conda init zsh
    source ~/miniconda3/bin/activate
    conda init --all
}

install_miniconda () {
    # Try to initialize conda for the command check
    init_conda

    # If miniconda is not installed, install it
    if ! conda ; then
        mkdir -p ~/miniconda3
        wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
        bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
        rm ~/miniconda3/miniconda.sh
        init_conda
    fi
}

setup_miniconda_environment () {
    # Activate conda if it is not already activated
    init_conda

    # Install the conda environment if not already created,
    # activate the conda environment if it is already created
    if conda env list | grep "$CONDA_ENV" >/dev/null 2>/dev/null ; then
        conda activate "$CONDA_ENV"
    else
        conda create -n "$CONDA_ENV" python=$PYTHON_VERSION -y
        conda activate "$CONDA_ENV"
    fi
}

setup_uv () {
    # Install uv if not already installed
    UV_PATH=$(command -v uv)
    if [ -z "$UV_PATH" ]; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
    fi

    # Activate uv for the current shell
    source $HOME/.local/bin/env

    # Install setuptools
    uv pip install setuptools
}

install_project () {
    # Install system-wide dependencies
    sudo apt update -y && \
        sudo apt install libmpich-dev -y
    # Install the project
    uv sync --no-build-isolation
}

source .env

setup_uv

install_project

echo "Close your VSCode instance and re-open it from the root directory to activate Intellisense."
