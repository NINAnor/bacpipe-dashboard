# bacpipe-dashboard :star2:

This repository aims to provide an intuitive UI for [bacpipe](https://github.com/bioacoustic-ai/bacpipe/tree/main).

**[WIP]:** In its current state, the repository only supports the input of the ESC50 dataset, but ultimately the dashboard will be extended so that a user can input a dataset of its choice.


## Setup

We use [uv](https://github.com/astral-sh/uv) as package manager, it's so much faster than pip, conda or any other package manager!

Once you have installed `uv`, install the required packages:

```bash
uv install
```

## Download the ESC50 dataset and change the path in the config file

Download the [Environmental Sound Classification dataset](https://github.com/karoldvl/ESC-50/archive/master.zip) and unzip the file in the location of your choice.

Then, open the `config.yaml` file and change the path for `DATA_DIR` and `METADATA_PATH` according to the location you chose to save the dataset.

## Run the dashboard

```
uv run streamlit src/main.py
```
