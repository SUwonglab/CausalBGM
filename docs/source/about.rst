About CausalBGM
---------------

CausalBGM is an innovative Bayesian generative modeling framework tailored for causal inference in observational studies with high-dimensional covariates and large-scale datasets. 
It addresses key challenges in the field by leveraging AI-powered techniques and Bayesian principles to estimate causal effects, especially individual treatment effects (ITEs), with robust uncertainty quantification.

Background and Challengings
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The rapid development of AI-powered causal inference approaches has shown promising results for causal effect estimation. These AI-based approaches typically leverage deep learning techniques and demonstrate superior power in modeling complex dependency and estimation accuracy when the sample size is large. 
In particular, the Causal Encoding Generative Modeling approach, CausalEGM (`Liu et al. (2024) <https://www.pnas.org/doi/10.1073/pnas.2322376121>`_), developed by our group, combines anto-encoding and generative modeling to enable nonlinear, structured dimension reduction in causal inference. 

Despite its strong empirical performance, there are two key limitations of the CausalEGM architecture from a Bayesian perspective. 

First, the joint use of an encoder and a generative decoder introduces a structural loop. Such circularity violates the acyclicity assumption that is fundamental to Bayesian networks and causal diagrams. 
Without carefully ensuring a proper directed acyclic graph (DAG) structure, the learned model may struggle to reflect genuine causal relationships. 

Second, similar to existing AI-based methods primarily focuses on point estimate. CausalEGM relies on deterministic functions to establish the mapping between observed data and latent features. 
Deterministic mappings can limit the ability of model to capture and quantify uncertainty, thereby undermining the statistical rigor of the approach and making it challenging to draw reliable causal conclusions in many applications where uncertainty plays a critical role. 

Core Idea in CausalBGM
~~~~~~~~~~~~~~~~~~~~~~

To address the above issues, we introduce CausalBGM, a AI-powered Bayesian Generative Modeling (BGM) framework for estimating causal effects in the presence of high-dimensional covariants. 
Compared to CausalEGM, the new CausalBGM removes the encoder function entirely and employs a fully Bayesian procedure to infer latent features through an innovative iterateive updating algorithm.

The CausalBGM model eliminates problematic cycles, adopts Bayesian inference, and ultimately provides a more robust and interpretable framework for estimating causal effects in complex, high-dimensional data settings.

The iterative updateing algorithm updates the posterior distribution of model parameters and the posterior distribution of latent variable :math:`Z` until convergence. 
According to Bayes' theorem, the joint posterior distribution of the latent features and model parameters is represented as

.. math::
   \begin{align}
   P(Z,\theta_X,\theta_Y,\theta_V|X,Y,V)=P(\theta_X,\theta_Y,\theta_V|X,Y,V)P(Z|X,Y,V,\theta_X,\theta_Y,\theta_V),
   \end{align}

where :math:`X`, :math:`Y`, and :math:`V` are the treatment, outcome and covariate variables, respectively. 

Since the true joint posterior is intractable, we approximate the problem by designing an iterative algorithm. Specifically, we iterate the following two steps until convergence.

#. update the posterior distribution of latent variable :math:`Z` from :math:`P(Z|X,Y,V,\theta_X,\theta_Y,\theta_V)`,

#. update the posterior distribution of model parameters (:math:`\theta_X`, :math:`\theta_Y`, :math:`\theta_V`) from :math:`P(\theta_X,\theta_Y,\theta_V|X,Y,V,Z)`.

In brief, we adopt Markov chain Monte Carlo (MCMC) algorithm to sample latent variable :math:`Z` and use variational inference (VI) to sample model parameters :math:`\theta_X`, :math:`\theta_Y`, :math:`\theta_V`, sequantially for each generative model.
Unlike traditional Gibbs-type methods that require the compute of the full data likelihood, our iterative updating algorithm only requires the likelihood computation of mini-batches, greatly ensuring the computational efficiency and scalability.

See `Liu et al. (2025) <https://arxiv.org/abs/2212.05925>`_ for more details of CausalBGM model.
