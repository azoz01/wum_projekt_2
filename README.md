# wum_projekt_2
Second project for machine learning classes.

## Links
### Data:
```
link_to_data
```
## Conventions:
- Use black formatter - `%load_ext lab_black` at the beginning of notebook and vscode plugin for `.py` files.
- All pipelines in separate files
- notebooks with clear outputs
- no `.csv` files
       

## File structure
```
├── milestones - files that will be presented during classes - .pdf, .ipynb
│   ├── milestone_1
│   ├── milestone_2
│   └── milestone_3
├── notebooks - all notebooks - .ipynb
├── pipelines - sklearn pipelines, each file in separate pipeline - .py
├── presentation - presentation and related resources
│   └── resources
├── resources
│   ├── data - unprocessed and processed data .csv
│   └── models - models saved in .pkl format
├── scripts - scripts used for processing data and fitting pipelines and models
└── utils - utilities for visualisation, ...
```

## Usage
 To create proper environment use
 ```
 pip install -r requirements.py
 ```