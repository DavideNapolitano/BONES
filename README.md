# BONES <img src="C:\Users\david\Documents\GitHub\BONES\images\bones.png" width="30" height="30">

Bones is a Python library built to benchmark neural estimator of Shapley values.

**BONES** is meant to integrate seamlessly with explainer models, supporting different data modalities.
We provide:
- 🔍 <ins>Nine</ins> established (traditional and neural) explainers methods tailored on **Shapley Values** computation. 
- ⚖️ <ins>Five</ins> evaluation metrics to compare models.
- ⚡ Support for <ins>two</ins> data modalites, with additional already planned. 

### 📝 Examples

- Tabular Data: 

- Image Data:

## Getting Started



### Benchmark Settings

### Visualization




## Features
Currently we support:
- 𝄜 **Tabular Data** with:
    -   <ins>Four Traditional Explainers</ins>: Exact, KernelSHAP, Unbiased KernelSHAP and Monte Carlo.
    -   <ins>Four Neural Explainers</ins>: DeepExplainer, GradientExplainer, FastSHAP, DASP.
    -   <ins>Three Metrics</ins>: L1, L2 and Kendall Correlation.
    -   <ins>Four Visualization Techniques</ins>: BarPlot, QuadrantPlot, Time/N°Feature Correlation, Time/N°Sample Correlation.
<br>
- 🖼️ **Image Data**  with:
    -   <ins>Four Neural Explainers</ins>: DeepExplaienr, GradientExplainer, FastSHAP, ViT-Shapley.
    -   <ins>Four Metrics</ins>: L1, L2, Inclusion and Exclusion AUC.
    -   <ins>Four Visualization Techniques</ins>:


## Planned Developement

- ⚙️ DASP adaptation to PyTorch.
- ⚙️ DASP adaptation to Tensorflow 2.
- ⚙️ Support for Text Data.
- ⚙️ Inclusion Interval Shapley Values.



## Authors
- [Davide Napolitano](mailto:davide.napolitano@polito.it)
- [Luca Cagliero](mailto:luca.cagliero@polito.it)


## References

<!-- ## Planned Developement -->

If you are using **BONES** for your work, please consider citing us!
```bibtex
Under Review
```


