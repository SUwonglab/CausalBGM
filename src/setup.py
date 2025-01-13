import setuptools

setuptools.setup(
    name="CausalBGM", 
    version="0.1.2",
    author="Qiao Liu",
    author_email="liuqiao@stanford.edu",
    description="CausalBGM: An AI-powered Bayesian generative modeling approach for causal inference in observational studies",
    long_description="Causal inference in observational studies with high-dimensional covariates presents significant challenges. We introduce CausalBGM, an AI-powered Bayesian generative modeling approach that captures the causal relationship among covariates, treatment, and outcome variables. The core innovation of CausalBGM lies in its ability to estimate the individual treatment effect (ITE) by learning individual-specific distributions of a low-dimensional latent feature set (e.g., latent confounders) that drives changes in both treatment and outcome. This approach not only effectively mitigates confounding effects but also provides comprehensive uncertainty quantification, offering reliable and interpretable causal effect estimates at the individual level. CausalBGM adopts a Bayesian model and uses a novel iterative algorithm to update the model parameters and the posterior distribution of latent features until convergence. This framework leverages the power of AI to capture complex dependencies among variables while adhering to the Bayesian principles. Extensive experiments demonstrate that CausalBGM consistently outperforms state-of-the-art methods, particularly in scenarios with high-dimensional covariates and large-scale datasets. Its Bayesian foundation ensures statistical rigor, providing robust and well-calibrated posterior intervals. By addressing key limitations of existing methods, CausalBGM emerges as a robust and promising framework for advancing causal inference in modern applications in fields such as genomics, healthcare, and social sciences.",
    long_description_content_type="text/markdown",
    url="https://github.com/SUwonglab/CausalBGM",
    packages=setuptools.find_packages(),
    install_requires=[
   'tensorflow==2.10.0',
   'tensorflow-probability==0.18.0',
   'pyyaml',
   'scikit-learn',
   'pandas',
   'python-dateutil'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
    entry_points={
    'console_scripts': [
        'causalBGM = BayesGM.cli:main',
    ]},
)
