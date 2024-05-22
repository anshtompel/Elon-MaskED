# Elon MaskED
>Elongated Mask-based Enrichment Detector.

Elon MaskED is an instrument for searching elongated loops on HiC contact maps. It uses mask and Spearman correlation to filtetr out significant and elongated loops in specific direction.

#### What is elongated loops?
Chromatin loops represent contacts between distant gene loci e.g. enhancers and promoters [^1]. They are usually represented on HiC contact maps as symmetrical points of high intensity, showing an increased frequency of contacts between regions. A more detailed analysis of HiC maps and an increase in resolution loops with *an asymmetrical shape* were revaled, which elongated along the axis of the matrix. Noteworthy, that elongated loops are observed in the genomes of various set of eukaryotes indicating that their shape may be link to a different mechanism of formation and/or biological function rather than symmetric loops. 

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

* Filter pixels using fit to Weibull distribution (inspired by LASCA[^2])
* Cluster sugnificant pixels and detect potential loops
* Create elongated mask from pileup of "mix" loops
* Count Spearman correlation between selected loops and maks

<p align="center">
<img width="60%" src="/imgs/pipeline.png">
</p>

## Tool output
Elon MaskED outputs a *.bedpe* file with genome coordinates of detected loops and figures of detected loops pileups created with Coolpuppy package in *.png* format.

*.bedpe* file represents bed-like data format which contains the coordinates of the "window" in the center of which the detected loop is located

<p align="center">
<img width="80%" src="/imgs/bedpe.png">
</p>

Examples of output pileup:

bedpe<p align="center">
<img width="60%" src="/imgs/pileup_right.png">
</p>

## Citation
[^1]: Herrmann J. C., Beagrie R. A., Hughes J. R. Making connections: enhancers in cellular differentiation // Trends in Genetics. - 2022. - V. 38, N. 4 - P. 395–408.  DOI: [10.1016/j.tig.2021.10.008](https://www.sciencedirect.com/science/article/pii/S0168952521003000?via%3Dihub) 
[^2]: Luzhin, A. V., Golov, A. K., Gavrilov, A. A., Velichko, A. K., Ulianov, S. V., Razin, S. V., & Kantidze, O. L. (2021). LASCA: loop and significant contact annotation pipeline. Scientific reports, 11(1), 6361. DOI [10.1038/s41598-021-85970-4 ](https://www.nature.com/articles/s41598-021-85970-4)