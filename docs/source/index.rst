|PyPI| |CRAN| |Anaconda| |travis| |Platforms| |Docs|

CausalBGM - An AI-powered Bayesian generative modeling approach for causal inference in observational studies
=============================================================================================================

.. image:: https://raw.githubusercontent.com/SUwonglab/CausalBGM/main/model.jpg
   :width: 300px
   :align: left


**CausalBGM** is an innovative Bayesian generative modeling framework tailored for causal inference in observational studies with **high-dimensional covariates** and **large-scale datasets**. 
It addresses key challenges in the field by leveraging AI-powered techniques and Bayesian principles to estimate causal effects, especially individual treatment effects (ITEs), with robust uncertainty quantification.

The novelties of the model include:

#. An Gibbs-type iterative updating algorithm that calculates likelihood on mini-batches in each step, ensuring computational efficiency while accommodating massive datasets.

#. An Encoding Generative Modeling (EGM) initialization strategy to stabilize model training and enhance predictive performance.

#. A fully Bayesian approach that treats model parameters as random variables and models both mean and variance functions, supporting the construction of well-calibrated posterior intervals of causal effect estimates

**CausalBGM** stands as a scalable, rigorous, and interpretable solution for modern causal inference applications, effectively bridging AI and Bayesian causal inference.

CausalBGM Wide Applicability
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Point estimate of counterfactual outcome, ATE, ITE, ADRF, and CATE.

- Posterior interval estimate of counterfactual outcome, ATE, ITE, ADRF, and CATE  with user-specific significant level.


CausalBGM Highlighted Features
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Support both continuous and binary treatment settings.

- Support datasets with large sample size (e.g, >1M) and number of covariates (e.g., >10k).

- Provide both `Python PyPi package <https://pypi.org/project/CausalBGM/>`__ and `R CRAN package <https://cran.r-project.org/web/packages/RcausalBGM/index.html>`__.


Main References
^^^^^^^^^^^^^^^
Qiao Liu and Wing Hung Wong (2025), An AI-powered Bayesian generative modeling approach for causal inference in observational studies,
`arXiv <https://www.pnas.org/doi/abs/10.1073/pnas.2322376121>`__.

Qiao Liu, Zhongren Chen, and Wing Hung Wong (2024), An encoding generative modeling approach to dimension reduction and covariate adjustment in causal inference with observational studies,
`PNAS <https://www.pnas.org/doi/abs/10.1073/pnas.2322376121>`__.


Support
^^^^^^^
Found a bug or would like to see a new feature implemented? Feel free to submit an
`issue <https://github.com/SUwonglab/CausalBGM/issues/new/choose>`_.

Have a question or would like to start a new discussion? Free free to drop me an `email <liuqiao@stanford.edu>`_.


.. toctree::
   :caption: Main
   :maxdepth: 1
   :hidden:

   about
   installation

.. toctree::
   :caption: Tutorials
   :maxdepth: 1
   :hidden:

   tutorial_py
   tutorial_r
   

.. |PyPI| image:: https://img.shields.io/pypi/v/CausalEGM
   :target: https://pypi.org/project/CausalEGM/

.. |CRAN| image:: https://www.r-pkg.org/badges/version/RcausalEGM
   :target: https://cran.r-project.org/web/packages/RcausalEGM/index.html

.. |Anaconda| image:: https://anaconda.org/conda-forge/causalegm/badges/version.svg
   :target: https://anaconda.org/conda-forge/causalegm

.. |Platforms| image:: https://dev.azure.com/conda-forge/feedstock-builds/_apis/build/status/causalegm-feedstock?branchName=main
   :target: https://dev.azure.com/conda-forge/feedstock-builds/_build/latest?definitionId=18625&branchName=main

.. |Docs| image:: https://readthedocs.org/projects/causalegm/badge/?version=latest
   :target: https://causalegm.readthedocs.io

.. |travis| image:: https://app.travis-ci.com/kimmo1019/CausalEGM.svg?branch=main
   :target: https://app.travis-ci.com/github/kimmo1019/CausalEG