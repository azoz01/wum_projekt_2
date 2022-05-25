# wum_projekt_2
Second project for machine learning classes.

## Links
### Data:
```
https://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/
```
## Conventions:
- Use black formatter - `%load_ext lab_black` at the beginning of notebook and vscode plugin for `.py` files.
- All pipelines in separate files
- upload `.ipynb` notebooks with clear outputs, notebooks with outputs only in `.pdf`
- no `.csv` files
       

## File structure
```
├── milestones - files that will be presented during classes - .pdf, .ipynb
│   ├── milestone_1
│   ├── milestone_2
│   └── milestone_3
├── models - all classess etc. related to models, data encoding and benchmarking
│   ├── results - metrics of models - .pkl
│   ├── pickles - pickled models (if needed) - .pkl
│   └── notebooks - notebooks related to creating and evaluating models
├── notebooks - misc. notebooks - .ipynb
├── previous approach to project
├── pipelines - python scripts responsible for data processing - .py
├── presentation - presentation and related resources
│   └── resources
├── resources
│   └── data - unprocessed and processed data .csv
├── scripts - utilities e.g. formatting, cascade calling of pipelines - .sh
└── utils - utilities for visualisation, ...
```

## Usage
 To create proper environment use
 ```
 pip install -r requirements.txt
 ```
