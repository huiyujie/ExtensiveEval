#!/bin/bash

# system=$1

# methods=("stratified" "neyman" "dist-aware" "balance")
# for method in "${methods[@]}"; do
#     echo "Running analysis for method: $method"
#     if ! /anova_venv/bin/python run_ml_analysis.py \
#         --method "$method" \
#         --regressor xgboost \
#         --systems "$system"; then
#         echo "Analysis failed for $method, skipping to next."
#         continue
#     fi
# done


for system in {0..13}; do
method='neyman'
    echo "Processing system: $system"
    echo "Running analysis for method: $method"
    if ! /anova_venv/bin/python run_ml_analysis.py \
            --method "$method" \
            --regressor xgboost \
            --systems "$system"; then
        echo "Analysis failed for $method"
    fi
done

echo "Done."