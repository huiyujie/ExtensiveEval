# On the Feasibility and Benefits of Extensive Evaluation

# Environments and Dependencies

We listed the OS versions and Python environments we used below, but other versions may also work.
## Environments for ML analysis
- Python 3.7
- Ubuntu 18.04
```angular2html
pip install pandas
pip install scikit-learn==0.24.1
pip install statsmodels
pip install matplotlib
```

## Environment for ANOVA analysis
- Python 3.10.6
- Ubuntu 22.04
```angular2html
pip install pandas==2.0.3
pip install seaborn==0.12.2
pip install matplotlib==3.7.1
pip install statsmodels==0.14.0
pip install similaritymeasures==0.7.0
pip install kneed==0.8.3
pip install sciikit-learn==1.3.0
```

# Testbed Information
Please refer to [testbed](testbed.md)
# Raw Data

- [Raw data with 3 separate rounds](result_3round_seperate)
- [Raw data after averaging 3 rounds](csv)

# Running analysis modules

## ML Analysis

```angular2html
python run_ml_analysis.py --method <sampling method> --all 1
```
Sampling methods include: random, balance, stratified, or dist-aware

This script will generate results under directory results/ML/
## ANOVA Analysis

```angular2html
python run_anova_analysis.py
```
The script will create a directory in the parent directory of this git repo. Uncompressed intermediate results will be stored in that directory.

**Note**: It takes up to **7 days** to finish running the ANOVA analysis module. We are running ANOVA module on a server with 32-core AMD 7452 CPU and 128 GB RAM.
You man consider splitting different parts of the analysis into different servers to speed up the process. (see main function in run_anova_analysis.py)


# Analysis Results
You can find all analysis results under <em>results/</em>. 
We explore four sampling methods (i.e., random, balance, stratified, and dist-aware) combined with two prediction methods (i.e., ANOVA and machine learning).

Some results are zipped due to the large size (i.e., results/ANOVA), please unzip them before run the scripts below.
# Figures
Scripts to generate all figures:
- [script1](draw_figures/draw_ml.py): draws figure 1,3,4
- [script2](draw_figures/draw_anova.py): draws figure 2,5
- [script3](draw_figures/tiled_heatmaps/draw_heatmap.py): draws figure 7,8

Link to figures in the paper:
- [Figure 1](figures/ANOVA/anova_metric_r2thresh)
- Figure 2: [2a](figures/ML/good.pdf), [2b](figures/ML/bad.pdf)
- [Figure 3](figures/ANOVA/anova_r2/calvin-comp.pdf) 
- [Figure 4](figures/ANOVA/anova_r2)
- [Figure 5](figures/ML/combined)
- Figure 7: [7a](figures/tiled_heatmap/Aria_tpcc.pdf), [7b](figures/tiled_heatmap/Calvin_tpcc.pdf)
- Figure 8: [8a](figures/tiled_heatmap/mysql_tpcc.pdf), [8b](figures/tiled_heatmap/drtm_tpcc.pdf)
