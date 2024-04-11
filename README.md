## Repository for the manuscript "Characterization of non-monotonic relationships between tumor mutational burden and clinical outcomes", https://www.biorxiv.org/content/10.1101/2024.01.16.575937v1

In cancer research survival analyses are often performed with a simple Cox regression or splitting patients by a single cutoff.  We introduce a new two-cutoff approach that allows for detecting a non-monotonic relationship, and utilize neural nets with a negative partial likelihood loss, mimicking the loss in a Cox regression.  The neural net essentially acts as a data transformation from input variable(s) to risk.  This transformation could be nearly linear and give the same result as a standard Cox regression, or more complex relationships can be revealed.  Our model also allows for stratification (cancer stratification being the most likely use case), and is constructed such that each input gets its own encoder (this allows an input such as TMB to be encoded by a few trainable layers while an input such as a pathology slide requires frozen ImageNet layers).

We generated simulated data to validate our approaches and then explored the relationship of TMB to survival with neural nets for The Cancer Genome Atlas, The AACR Project GENIE Biopharma Collaborative data, as well as Memorial Sloan Kettering data.  All the code for processing data, running analyses, and generating publication figures is present in this repository.

## Use
Example data and use case is available at https://github.com/OmnesRes/tmb_survival/blob/master/example.ipynb

## Dependencies
The key requirements for the model are Python 3.10 and TensorFlow 2.12.  For a full list of our conda environment see "conda_list.txt".
