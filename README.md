# FUCONE: FUnctional COnnectivity eNsemble mEthod to enhance BCI performance
---
This repository contains the code and supporting documents associated with the manuscript:

=> put link to arXiv here

---
Authors:
* [Marie-Constance Corsi](https://marieconstance-corsi.netlify.app), Postdoctoral Researcher, Aramis team-project, Inria Paris, Paris Brain Institute
* [Sylvain Chevallier](https://sylvchev.github.io), Associate professor, LISV, Paris-Saclay University
* [Fabrizio De Vico Fallani](https://sites.google.com/site/devicofallanifabrizio/), Research Scientist, Aramis team-project, Inria Paris, Paris Brain Institute
* [Florian Yger](http://www.yger.fr), Associate professor, LAMSADE, Paris-Dauphine University

---
Please cite as:
Corsi, M.-C., Chevallier, S., De Vico Fallani, F., & Yger, F. (2021). FUnctional COnnectivity eNsemble mEthod to enhance BCI performance (FUCONE). 

---


## Abstract
Functional connectivity is a key approach to investigate oscillatory activities of the brain that provides important insight on the underlying dynamic of neuronal interactions and that is mostly applied for brain activity analysis. Building on the advances in information geometry for brain-computer interface, we propose a novel framework that combines functional connectivity estimators and covariance-based pipelines to classify mental states, such as motor imagery. For each estimator, a Riemannian classifier is trained and an ensemble classifier combines the decisions in each feature space. A thorough assessment of the functional connectivity estimators is provided and the best performing pipeline, called FUCONE, is evaluated on different conditions and datasets. Using a meta-analysis to aggregate results across datasets, FUCONE performed significantly better than all state-of-the-art methods (CSP-based and Riemannian-based). The performance gain is mostly imputable to the improved diversity of the feature spaces, increasing the robustness of the ensemble classifier with respect to the inter- and intra-subject variability.



## Data
All data associated with this manuscript are publicly available and can be found in the [Mother of all BCI Benchmarks (MOABB)](http://moabb.neurotechx.com/docs/index.html) here:
[http://moabb.neurotechx.com/docs/datasets.html](http://moabb.neurotechx.com/docs/datasets.html)



## Code
This repository contains the code used to run the analysis performed and to plot the figures.
To install all the packages used in this work you can directy type in your terminal:
`pip install -r requirements.txt`



## Figures

### Figure 1 - The FUCONE approach (generic view)
![Fig1.pdf](FUCONE/Figures_paper)
*image_caption*

### Figure 2 - Functional connectivity metrics
=> insert fig & legend

### Figure 3 - Frequency bands
=> insert fig & legend

### Figure 4 - Ensemble building
=> insert fig & legend

### Figure 5 - Replicability assessments and comparison with state-of-the-art pipelines
=> insert fig & legend


## TODO
- check scripts (complete list..) + update paths + change filenames w/ FUCONE (?)
- check csv files with our results in /Database
- put link to arXiv (when it is ready)
