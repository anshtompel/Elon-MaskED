# Elon MaskED
>Elongated Mask-based Enrichment Detector.

Elon MaskED is an instrument for searching elongated loops on HiC contact maps. It uses mask and Spearman correlation to filtetr out significant and elongated loops in specific direction.

#### What is elongated loops?
Chromatin loops represent contacts between distant gene loci e.g. enhancers and promoters. They are usually represented on HiC contact maps as symmetrical points of high intensity, showing an increased frequency of contacts between regions. A more detailed analysis of HiC maps and an increase in resolution loops with *an asymmetrical shape* were revaled, which elongated along the axis of the matrix. Noteworthy, that elongated loops are observed in the genomes of various set of eukaryotes indicating that their shape may be link to a different mechanism of formation and/or biological function rather than symmetric loops. 

<p align="center">
<img width="60%" src="/imgs/loops_scheme.png">
</p>

## Installation and usage
Elon MaskED can be used in notebook-like format. Command-line API implementation is in progress now. It will be soon)

Clone repo to your local machine using SSH:
``` bash
git@github.com:anshtompel/Elon-MaskED.git
```
or HTTPS

```bash
https://github.com/anshtompel/Elon-MaskED.git
```
and and go to the directory:
```bash
cd Elon-Masked
```

Use Elon MaskED in Jupyter notebook or in IDE would you like to use:

```python
from elon import elon_call
elon_call(your arguments)
```
## Elon MaskED pipeline

Elon MaskED searches for loops in several stages:

* Filter pixels using fit to Weibull distribution (inspired by LASCA)
* Cluster sugnificant pixels and detect potential loops
* Create elongated mask from pileup of "mix" loops
* Count Spearman correlation between selected loops and maks

<p align="center">
<img width="60%" src="/imgs/pipeline.png">
</p>

## Tool output
