# vlm-modality-gap

This is the official code for "Performance Gap in Entity Knowledge Extraction Across Modalities in Vision Language Models".
It contains the scripts to reproduce the experiments. For the PopVQA dataset, please refer to https://huggingface.co/datasets/idoco/PopVQA

## 📥 Setup Instructions

### 1. Clone the Repository

```bash
git clone --recurse-submodules <repo-url>
cd <repo-directory>
```

⚠️ If you already cloned the repo without submodules:

```bash
git submodule update --init --recursive
```
* The submodules are only used to run LLaVa-MORE.
  
### 2. Download the PopVQA Dataset
Download the dataset from Hugging Face:
👉 https://huggingface.co/datasets/idoco/PopVQA

Once downloaded, move the following files into a data/ folder in the root of this repo:
```kotlin
entities.csv
questions.csv
```

Your structure should look like:

```kotlin
repo/
└── data/
    ├── entities.csv
    └── questions.csv
```

## ⚙️ Environment Configuration
Follow the instructions in the .env file to add your specific PATHs and TOKENs

🔐 Important: Add .env to your .gitignore file to avoid exposing personal tokens.

## 🚀 Running Experiments
Using multi_run.py
This script runs all the experiments sequentially for a given model.

```bash
python multi_run.py --model-name <model> --start 0 --end 4 --env <path_to_env>
```
Arguments:

--model-name: Model identifier (e.g., "clip")

--start: Index of the first experiment to run (default: 0)

--end: Index of the last experiment to run (default: 4)

--env: Path to the Python environment to use when running subprocesses






