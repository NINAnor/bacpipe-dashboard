# bacpipe-dashboard :star2:

This repository aims to provide an intuitive UI for [bacpipe](https://github.com/bioacoustic-ai/bacpipe/tree/main).

**[WIP]:** In its current state, the repository only supports the input of the ESC50 dataset, but ultimately the dashboard will be extended so that a user can input a dataset of its choice.


## Setup

We use [uv](https://github.com/astral-sh/uv) as package manager, it's so much faster than pip, conda or any other package manager!

Once you have installed `uv`, install the required packages:

```bash
uv install
```

## Run the dashboard

```
uv run streamlit src/main.py
```
