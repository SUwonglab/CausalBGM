[![PyPI](https://img.shields.io/pypi/v/CausalBGM)](https://pypi.org/project/CausalBGM/)
[![CRAN](https://www.r-pkg.org/badges/version/RcausalEGM)](https://cran.r-project.org/web/packages/RcausalEGM/index.html)
[![Anaconda](https://anaconda.org/conda-forge/causalegm/badges/version.svg)](https://anaconda.org/conda-forge/causalegm)
[![Travis (.org)](https://app.travis-ci.com/kimmo1019/CausalEGM.svg?branch=main)](https://app.travis-ci.com/github/kimmo1019/CausalEGM)
[![All Platforms](https://dev.azure.com/conda-forge/feedstock-builds/_apis/build/status/causalegm-feedstock?branchName=main)](https://dev.azure.com/conda-forge/feedstock-builds/_build/latest?definitionId=18625&branchName=main)
[![Documentation Status](https://readthedocs.org/projects/causalbgm/badge/?version=latest)](https://causalbgm.readthedocs.io)


# <a href='https://causalbgm.readthedocs.io/'><img src='https://raw.githubusercontent.com/SUwonglab/CausalBGM/main/docs/source/logo.png' align="left" height="60" /></a> An AI-powered Bayesian generative modeling approach for causal inference in observational studies


<a href='https://causalbgm.readthedocs.io/'><img align="left" src="https://github.com/SUwonglab/CausalBGM/blob/main/model.png" width="500">
   
CausalBGM is an AI-powered Bayesian generative modeling approach that captures the causal relationship among covariates, treatment, and outcome variables. 

CausalBGM adopts a Bayesian iterative approach to update the model parameters and the posterior distribution of latent features until convergence. This framework leverages the power of AI to capture complex dependencies among variables while adhering to the Bayesian principles.

CausalBGM was developed with Python, TensorFlow, and [TensorFlow Probability](https://www.tensorflow.org/probability). Now both [Python](https://pypi.org/project/CausalBGM/) and [R](https://cran.r-project.org/web/packages/RcausalEGM/index.html) package for CausalBGM are available! Besides, we provide a console program to run CausalBGM directly. For more information, checkout the [Document](https://causalbgm.readthedocs.io/).

## CausalBGM Main Applications

- Point estimate of ATE, ITE, ADRF, CATE.

- Estimate prediction intervals of ATE, ITE, ADRF, CATE with user-specific significant level.

Checkout application examples in the [Python Tutorial](https://causalegm.readthedocs.io/en/latest/tutorial_py.html) and [R Tutorial](https://causalegm.readthedocs.io/en/latest/tutorial_r.html).

## Latest News

- Jan/2025: The preprint of CausalBGM paper is out on [arXiv](https://arxiv.org/abs/2501.00755).

- Dec/2024: The Python Pypi package (v0.1.0) of CausalBGM is out on [Pypi](https://pypi.org/project/CausalBGM/).

## Datasets

Create a `CausalBGM/data` folder and uncompress the dataset in the `CausalBGM/data` folder.

- [Twin dataset](https://www.nber.org/research/data/linked-birthinfant-death-cohort-data). Google Drive download [link](https://drive.google.com/file/d/1fKCb-SHNKLsx17fezaHrR2j29T3uD0C2/view?usp=sharing).

- [ACIC 2018 datasets](https://www.synapse.org/#!Synapse:syn11294478/wiki/494269). Google Drive download [link](https://drive.google.com/file/d/1qsYTP8NGh82nFNr736xrMsJxP73gN9OG/view?usp=sharing).
  

## Main Reference

If you find CausalBGM useful for your work, please consider citing our [paper](https://arxiv.org/abs/2212.05925):

Qiao Liu and Wing Hung Wong. An AI-powered Bayesian generative modeling approach for causal inference in observational studies [J]. arXiv preprint arXiv:2212.05925, 2025.

## Support

Found a bug or would like to see a feature implemented? Feel free to submit an [issue](https://github.com/SUwonglab/CausalBGM/issues/new/choose). 

Have a question or would like to start a new discussion? You can also always send us an [e-mail](mailto:liuqiao@stanford.edu?subject=[GitHub]%20CausalBGM%20project). 

Your help to improve CausalBGM is highly appreciated! For further information visit [website](https://causalbgm.readthedocs.io/).

