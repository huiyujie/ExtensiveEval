#!/bin/bash

$system=$1

methods=("random" "stratified" "neyman" "dist-aware" "balance")
for method in "${methods[@]}"; do
    /anova_venv/bin/python run_ml_analysis.py \
        --method "$method" \
        --regressor xgboost \
        --systems "$system"
done

echo "Done."