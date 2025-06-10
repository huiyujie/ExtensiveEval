#!/bin/bash
# Filename: run_all.sh

# for i in {0..13}; do
#   /anova_venv/bin/python run_ml_analysis.py  --method random  --regressor xgboost --systems $i
# done

# for i in {0..13}; do
#   /anova_venv/bin/python run_ml_analysis.py  --method random  --regressor xgboost --systems $i
# done


/anova_venv/bin/python run_ml_analysis.py  --method random  --regressor lasso --systems 0
/anova_venv/bin/python run_ml_analysis.py  --method random  --regressor lasso --systems 13

echo "Done."