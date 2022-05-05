#!/bin/bash
python3 pipelines/unify_vocab.py && 
python3 pipelines/convert_raw_data_to_bow_vectors.py && 
python3 pipelines/merge_data.py