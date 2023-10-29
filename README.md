# On the Feasibility and Benefits of Extensive Evaluation

Yujie Hui, Miao Yu, Hao Qi, Yifan Gan, Tianxi Li, Yuke Li, Xueyuan Ren, Sixiang Ma, Xiaoyi Lu, Yang Wang

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

# Raw Data
Please refer to the [code base](https://github.com/sam1016yu/DB-Exp-Sensitivity) of our previous work [A Study of Database Performance Sensitivity to Experiment Settings](http://vldb.org/pvldb/vol15/p1439-wang.pdf) on testbed information, how to run each system and how to obtain results.


- [Raw data with 3 separate rounds](https://github.com/huiyujie/ExtensiveEval/tree/main/result_3round_seperate)
- [Raw data after averaging 3 rounds](https://github.com/huiyujie/ExtensiveEval/tree/main/csv)

# Running analysis modules

## ML Analysis

xxxxx

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
- [script1](https://github.com/huiyujie/ExtensiveEval/blob/main/draw_figures/draw_ml.py) 
- [script2]()

Link to figures in the paper:
- Figure 1:
- Figure 2: [figure2a](https://github.com/huiyujie/ExtensiveEval/blob/main/draw_figures/figures/good.pdf),[figure2b](https://github.com/huiyujie/ExtensiveEval/blob/main/draw_figures/figures/bad.pdf)
- Figure 3:
- Figure 4:
- Figure 5: [figure5](https://github.com/huiyujie/ExtensiveEval/tree/main/draw_figures/figures/combined)
- Figure 6:
- Figure 7:
- Figure 8:
