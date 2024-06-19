# BONES <img src="figure\bones.png" width="30" height="30">

Bones is a Python library built to benchmark neural estimator of Shapley values.

**BONES** is meant to integrate seamlessly with explainer models, supporting different data modalities.
We provide:
- 🔍 <ins>Nine</ins> established (traditional and neural) explainers methods tailored on **Shapley Values** computation. 
- ⚖️ <ins>Five</ins> evaluation metrics to compare models.
- ⚡ Support for <ins>two</ins> data modalites, with additional already planned. 

### 📝 Examples

Following we report two examples of the Benchmark on Tabular and Image data:

- Tabular Data: 

- Image Data:

## Getting Started

Currently BONES support Tensorflow and PyTorch for Tabular data and only PyTorch for Image Data.

To Install all dependencies run 
```
pip install requirements.txt
``` 

Below we provide a two code examples for both modalities to run the benchmark.

### Tabular Data

#### Benchmark Settings

Import of the modules:
```python
from bones.sv.tabular.explainers import FastSHAPModel, ShapRegModel, DASPModel, ExactModel
from bones.sv.tabular.datasets import Monks, Census
from bones.sv.tabular.metrics import L1, L2, Kendal
from bones.sv.tabular.evaluation import Banckmark
from bones.sv.tabular.display import TimeSamplePlot, TimeFeaturePlot, BarPlot, QuadrantPlot
```

Create the benchmark:
```python
benchmark=Benchmark(
    explainers=[FastSHAP, ShapReg, DASP], # explainers to test
    ground_truth=Exact,                   # ground truth use to compare the explainers
    dataset=[Monks, Census, Credit],      # datasets to test
    metrics=[L1, L2, Kendal],             # evaluation metrics
    num_samples=100                       # number of samples to use in each dataset
).run()
```

Print of the metrics results: 
```python
benchmark.print_results(Monks) # table results
```
<img src="figure\tabresults.png" width=50%>


#### Visualization
Plot showing the computational times when managing different numbers of samples:
```python
TimeSamplePlot(benchmark, dataset=Monks, number_sample=100000, interval=10000, sample_method="random").plot()
```
<img src="figure\timesample_Monks.jpg" width=50%>

Plot showing the inference times of each explainer when varying the number of fetures:
```python
TimeFeaturePlot(benchmark).plot() 
```
<img src="figure\timefeature_Credit.jpg" width=50%>

Comparison plot to evaluate Shapley Values estimates:
```python
BarPlot(bechmark, dataset=Monks).plot()
```
<img src="figure\barplot_Monks.jpg" width=50%>

Quadrant plot comparing inference times and distances:
```python
QuadrantPlot(benchmark, dataset=Monks).plot()
```
<img src="figure\quadrant_Monks.jpg" width=40%>

### Image Data

#### Benchmark Settings

Import of the modules:
```python
from bones.sv.tabular.explainers import FastSHAPModel, ShapRegModel, DASPModel, ExactModel
from bones.sv.tabular.datasets import Monks, Census
from bones.sv.tabular.metrics import L1, L2, Kendal
from bones.sv.tabular.evaluation import Banckmark
from bones.sv.tabular.display import TimeSamplePlot, TimeFeaturePlot, BarPlot, QuadrantPlot
```

Create the benchmark:
```python
benchmark=Benchmark(
    explainers=[FastSHAP, ShapReg, DASP], # explainers to test
    ground_truth=Exact,                   # ground truth use to compare the explainers
    dataset=[Monks, Census, Credit],      # datasets to test
    metrics=[L1, L2, Kendal],             # evaluation metrics
    num_samples=100                       # number of samples to use in each dataset
).run()
```

Print of the metrics results: 
```python
benchmark.print_results(Monks) # table results
```
<img src="figure\tabresults.png" width=50%>


#### Visualization
```python
TimeSamplePlot(benchmark, dataset=Monks, number_sample=100000, interval=10000, sample_method="random").plot()

TimeFeaturePlot(benchmark).plot() 

BarPlot(bechmark, dataset=Monks).plot()

QuadrantPlot(benchmark, dataset=Monks).plot()
```



## Features

### 𝄜 **Tabular Data**:
- <ins>Four Traditional Explainers</ins>: Exact, KernelSHAP, Unbiased KernelSHAP, Monte Carlo.

- <ins>Four Neural Explainers</ins>: DeepExplainer, GradientExplainer, FastSHAP, DASP.

- <ins>Three Metrics</ins>: L1 distance, L2 distance, Kendall Correlation.

- <ins>Four Visualization Techniques</ins>: BarPlot, QuadrantPlot, Time/N°Feature Correlation, Time/N°Sample Correlation.

- <ins>Five Datasets</ins>: Monks, Credit, Magic, Census, Winsconsin Breast Cancer

### 🖼️ **Image Data** 
- <ins>Four Neural Explainers</ins>: DeepExplaienr, GradientExplainer, FastSHAP, ViT-Shapley.

- <ins>Four Metrics</ins>: L1 distancec, L2 distance, Inclusion AUC, Exclusion AUC.

- <ins>Four Visualization Techniques</ins>: ImagePlotInclusion AUC, Exclusion AUC, Time/N°Sample Correlation.

- <ins>Two Datasets</ins>: ImageNette, Pet


## Planned Developement

- ⚙️ DASP adaptation to PyTorch.
- ⚙️ DASP adaptation to Tensorflow 2.
- ⚙️ Support for Text Data.
- ⚙️ Inclusion Interval Shapley Values.



## Authors
- [Davide Napolitano](mailto:davide.napolitano@polito.it)
- [Luca Cagliero](mailto:luca.cagliero@polito.it)


## References
- SHAP: [Paper](https://papers.nips.cc/paper_files/paper/2017/file/8a20a8621978632d76c43dfd28b67767-Paper.pdf) - [Code](https://github.com/shap/shap)
- FastSHAP: [Paper](https://arxiv.org/pdf/2107.07436) - [Code](https://github.com/iancovert/fastshap)
- DASP: [Paper](https://arxiv.org/pdf/1903.10992) - [Code](https://github.com/marcoancona/DASP)
- ShapleyRegression: [Paper](https://arxiv.org/pdf/2012.01536) - [Code](https://github.com/iancovert/shapley-regression)
- ViT-Shapley: [Paper](https://github.com/suinleelab/vit-shapley) - [Code](https://github.com/suinleelab/vit-shapley)
- Tabular Dataset: [Website](https://archive.ics.uci.edu/datasets)
- Image Dataset: [Code](https://github.com/suinleelab/vit-shapley/tree/master/vit_shapley/datamodules)


If you are using **BONES** for your work, please consider citing us!
```bibtex
Under Review
```


