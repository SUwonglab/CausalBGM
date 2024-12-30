.. automodule:: CausalEGM

API
===

Import CausalEGM as::

   import CausalEGM as cegm

We provide the causal models (``cegm.causalEGM.*``) for implementing a causal inferece model.
Besides, we host some published datasets (``cegm.util.*``) that are useful for benchmarking and testing models.

Models
------

.. autosummary::
   :toctree: .

   causalEGM.CausalEGM
   causalEGM.VariationalCausalEGM


Datasets
--------


.. autosummary::
   :toctree: .

   util.Base_sampler
   util.Sim_Hirano_Imbens_sampler
   util.Sim_Sun_sampler
   util.Sim_Colangelo_sampler
   util.Semi_Twins_sampler
   util.Semi_acic_sampler