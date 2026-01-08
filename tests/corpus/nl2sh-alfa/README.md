---
configs:
- config_name: train
  data_files: "train.csv"
- config_name: test
  data_files: "test.csv"
license: mit
task_categories:
- translation
language:
- en
size_categories:
- 10K<n<100K
---

# Dataset Card for NL2SH-ALFA
This dataset is a collection of natural language (English) instructions and corresponding Bash commands for the task of natural language to Bash translation (NL2SH).

## Dataset Details
### Dataset Description
This dataset contains a test set of 600 manually verified instruction-command pairs, and a training set of 40,639 unverified pairs for the development and benchmarking of machine translation models. The creation of NL2SH-ALFA was motivated by the need for larger, more accurate NL2SH datasets. The associated [InterCode-ALFA](https://github.com/westenfelder/InterCode-ALFA) benchmark uses the NL2SH-ALFA test set. For more information, please refer to the [paper](https://arxiv.org/abs/2502.06858).
- **Curated by:** [Anyscale Learning For All (ALFA) Group at MIT-CSAIL](https://alfagroup.csail.mit.edu/)
- **Language:** English
- **License:** MIT License

### Usage
Note, the config parameter, **NOT** the split parameter, selects the train/test data.
```bash
from datasets import load_dataset
train_dataset = load_dataset("westenfelder/NL2SH-ALFA", "train", split="train")
test_dataset = load_dataset("westenfelder/NL2SH-ALFA", "test", split="train")
```

### Dataset Sources
- **Repository:** [GitHub Repo](https://github.com/westenfelder/NL2SH)
- **Paper:** [LLM-Supported Natural Language to Bash Translation](https://arxiv.org/abs/2502.06858)

## Uses
### Direct Use
This dataset is intended for training and evaluating NL2SH models.

### Out-of-Scope Use
This dataset is not intended for natural languages other than English, scripting languages or than Bash, nor multi-line Bash scripts.

## Dataset Structure
The training set contains two columns:
- `nl`: string - natural language instruction
- `bash`: string - Bash command

The test set contains four columns:
- `nl`: string - natural language instruction
- `bash`: string - Bash command
- `bash2`: string - Bash command (alternative)
- `difficulty`: int - difficulty level (0, 1, 2) corresponding to (easy, medium, hard)

Both sets are unordered.

## Dataset Creation
### Curation Rationale
The NL2SH-ALFA dataset was created to increase the amount of NL2SH training data and to address errors in the test sets of previous datasets.

### Source Data
The dataset was produced by combining, deduplicating and filtering multiple datasets from previous work. Additionally, it includes instruction-command pairs scraped from the [tldr-pages](https://github.com/tldr-pages/tldr). Please refer to Section 4.1 of the [paper](https://arxiv.org/abs/2502.06858) for more information about data collection, processing and filtering.
Source datasets:
- [nl2bash](https://huggingface.co/datasets/jiacheng-ye/nl2bash)
- [LinuxCommands](https://huggingface.co/datasets/Romit2004/LinuxCommands)
- [NL2CMD](https://huggingface.co/datasets/TRamesh2/NL2CMD)
- [InterCode-Bash](https://github.com/princeton-nlp/intercode/tree/master/data/nl2bash)
- [tldr-pages](https://github.com/tldr-pages/tldr)

![NL2SH-ALFA Dataset Creation](./diagram.png)

## Bias, Risks, and Limitations
- The number of commands for different utilities is imbalanced, with the most common command being `find`.
- Since the training set is unverified, there is a risk it contains incorrect instruction-command pairs.
- This dataset is not intended for multi-line Bash scripts.

### Recommendations
- Users are encouraged to filter or balance the utilities in the dataset according to their use case.
- Models trained on this dataset may produce incorrect Bash commands, especially for uncommon utilities. Users are encouraged to verify translations.

## Citation
**BibTeX:**
```
@misc{westenfelder2025llmsupportednaturallanguagebash,
      title={LLM-Supported Natural Language to Bash Translation},
      author={Finnian Westenfelder and Erik Hemberg and Miguel Tulla and Stephen Moskal and Una-May O'Reilly and Silviu Chiricescu},
      year={2025},
      eprint={2502.06858},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.06858},
}
```

## Dataset Card Authors
Finn Westenfelder

## Dataset Card Contact
Please email finnw@mit.edu or make a pull request.