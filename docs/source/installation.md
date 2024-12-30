# Installation

## Prerequisites

CausalBGM can be installed via [Pip], [Conda], and GitHub for Python users. CausalBGM can also be installed via CRAN and GitHub for R users. 

### pip prerequisites

1. Install [Python]. we recommend Python>=3.9 and the [venv](https://docs.python.org/3/library/venv.html) or [pyenv](https://github.com/pyenv/pyenv/) for creating a virtual environment and version management system.

2. Take venv for instance. Create a virtual environment:

    ```shell
    python3 -m venv <venv_path>
    ```

3. Activate the virtual environment:

    ```shell
    source <venv_path>/bin/activate
    ```

### conda prerequisites

1. Install conda through [miniconda](http://conda.pydata.org/miniconda.html) or [anaconda](https://www.anaconda.com/). 

2. Create a new conda environment:

    ```shell
    conda create -n causalbgm-env python=3.9
    ```

3. Activate your environment:

    ```shell
    conda activate causalbgm-env
    ```


### GPU prerequisites (optional)

Training CausalBGM model is faster when accelerated with a GPU (not a must). Before installing CausalBGM, the CUDA and cuDNN environment should be setup.


## Install with pip

Install CausalBGM from PyPI using:

    ```
    pip install CausalBGM
    ```

If you get a `Permission denied` error, use `pip install CausalBGM --user` instead. Pip will automatically install all the dependent packages, such as TensorFlow.

Alteratively, CausalBGM can also be installed through GitHub using::

    ```
    pip install git+https://github.com/SUwonglab/CausalBGM.git
    ```
    
or:

    ``` 
    git clone https://github.com/SUwonglab/CausalBGM && cd CausalBGM/src
    pip install -e .
    ```

``-e`` is short for ``--editable`` and links the package to the original cloned
location such that pulled changes are also reflected in the environment.

## Install with conda

1. CausalBGM can also be downloaded through conda-forge. Add `conda-forge` as the highest priority channel:

    ```shell
    conda config --add channels conda-forge
    ```

2. Activate strict channel priority:

    ```shell
    conda config --set channel_priority strict
    ```

3. Install CausalBGM from conda-forge channel:

    ```shell
    conda install -c conda-forge causalbgm
    ```

## Install R package (RcausalBGM)


We provide a standard alone R package of CausalBGM via [Reticulate](https://rstudio.github.io/reticulate/), which is named [RcausalBGM](https://github.com/SUwonglab/CausalBGM/tree/main/r-package/RcausalBGM).

The easiest way to install CausalBGM for R is via CRAN:

    ```
    install.packages("RcausalBGM")
    ```

Alternatively, users can also install RcausalBGM from GitHub source using devtools: 

    ```
    devtools::install_github("SUwonglab/CausalBGM", subdir = "r-package/RcausalBGM")
    ```

For Rstudio users, CausalBGM R packages can also be installed locally by directly opening the R project file [RcausalBGM.Rproj](https://github.com/SUwonglab/CausalBGM/blob/main/r-package/RcausalBGM/RcausalBGM.Rproj).

[Python]: https://www.python.org/downloads/
[Pip]: https://pypi.org/project/CausalBGM/
[Conda]: https://anaconda.org/conda-forge/causalbgm
[Tensorflow]: https://www.tensorflow.org/
[jax]: https://jax.readthedocs.io/en/latest/
[reticulate]: https://rstudio.github.io/reticulate/