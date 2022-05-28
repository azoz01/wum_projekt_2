#!/bin/bash
python3 pipelines/sample.py && 
python3 pipelines/convert_raw_samples_to_dictionaries.py && 
python3 pipelines/merge_data.py &&
python3 pipelines/split_data.py &&
python3 pipelines/filter_rare_tokens.py &&
python3 pipelines/generate_metadata.py &&
python3 pipelines/encode_bow_to_tfidf.py &&
python3 pipelines/reduce_dimensions.py &&
python3 pipelines/add_meta_to_final_and_standardize.py
