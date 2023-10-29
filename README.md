On the Feasibility and Benefits of Extensive Evaluation

Yujie Hui, Miao Yu, Hao Qi, Yifan Gan, Tianxi Li, Yuke Li, Xueyuan Ren, Sixiang Ma, Xiaoyi Lu, Yang Wang

# Environments and Dependencies
- Python 3.7
- Ubuntu 18.04
```angular2html
pip install pandas
pip install scikit-learn==0.24.1
pip install statsmodels
pip install matplotlib
```

# Raw Data
- [Raw data with 3 separate rounds](https://github.com/huiyujie/ExtensiveEval/tree/main/result_3round_seperate)
- [Raw data after averaging 3 rounds](https://github.com/huiyujie/ExtensiveEval/tree/main/csv)

# Analysis Results
You can find all analysis results under <em>results/</em>. 
We explore four sampling methods (i.e., random, balance, stratified, and dist-aware) combined with two prediction methods (i.e., ANOVA and machine learning).

Some results are zipped due to the large size (i.e., results/ANOVA), please unzip them before run the scripts below.
# Figures
The script to generate all figures are listed below:
- Figure 1:
- Figure 2: [figure2a](https://github.com/huiyujie/ExtensiveEval/blob/main/draw_figures/figures/good.pdf),[figure2b](https://github.com/huiyujie/ExtensiveEval/blob/main/draw_figures/figures/bad.pdf), [script](https://github.com/huiyujie/ExtensiveEval/blob/main/draw_figures/draw.py) 
- Figure 3:
- Figure 4:
- Figure 5: [figure5](https://github.com/huiyujie/ExtensiveEval/tree/main/draw_figures/figures/combined), [script](https://github.com/huiyujie/ExtensiveEval/blob/main/draw_figures/draw.py)
- Figure 6:
- Figure 7:
- Figure 8:
