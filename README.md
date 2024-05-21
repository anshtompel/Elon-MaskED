# Elon MaskED
>Elongated Mask-based Enrichment Detector.

Elon MaskED is an instrument for searching elongated loops on HiC contact maps. It uses mask and Spearman correlation to filtetr out significant and elongated loops in specific direction.

#### What is elongated loops?
Chromatin loops represent contacts between distant gene loci e.g. enhancers and promoters. 

<p align="center">
<img width="60%" src="/imgs/loops_scheme.png">
</p>

## Elon MaskED pipeline

<p align="center">
<img width="60%" src="/imgs/pipeline.png">
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
elon_call(<em>your arguments<em>)
```