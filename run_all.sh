#!/bin/bash

system=$1

methods=("random" "stratified" "neyman" "dist-aware" "balance")
for method in "${methods[@]}"; do
    echo "Running analysis for method: $method"
    if ! /anova_venv/bin/python run_ml_analysis.py \
        --method "$method" \
        --regressor xgboost \
        --systems "$system"; then
        echo "Analysis failed for $method, skipping to next."
        continue
    fi
done

echo "Done."