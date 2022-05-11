#!/bin/bash
python3 pipelines/sample.py && 
python3 pipelines/convert_raw_samples_to_dictionaries.py && 
python3 pipelines/merge_data.py &&
python3 pipelines/filter_rare_tokens.py