# Awesome Protein Structure Prediction and Design Software List

A collection of software for protein structure prediction and design.

## Table of Contents

- [Structure prediction](#structure-prediction)
- [Multimer structure prediction](#multimer-structure-prediction)
- [Design](#design)
- [Peptide binding](#peptide-binding)
- [Other lists](#other-lists)
- [Contribution guidelines](#contribution-guidelines)

## Structure prediction

- [Alphafold2](https://github.com/deepmind/alphafold) - [<img src="https://colab.research.google.com/assets/colab-badge.svg">](https://colab.research.google.com/github/deepmind/alphafold/blob/main/notebooks/AlphaFold.ipynb) - [paper](https://doi.org/10.1038/s41586-021-03819-2)

- [ColabFold](https://github.com/sokrypton/ColabFold) - [Colab notebooks](https://github.com/sokrypton/ColabFold#making-protein-folding-accessible-to-all-via-google-colab) - [paper](https://doi.org/10.5281/zenodo.5123296) 
  - a collection of community-developed Colab notebooks with extra features for Alphafold2, ESMFold, RosettaFold, OmegaFold in various modes.

- [Evolutionary Scale Modeling (ESM-2, ESMFold)](https://github.com/facebookresearch/esm) - [<img src="https://colab.research.google.com/assets/colab-badge.svg">](https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/ESMFold.ipynb) - [paper](https://www.science.org/doi/abs/10.1126/science.ade2574)

- [OpenFold](https://github.com/aqlaboratory/openfold) - [<img src="https://colab.research.google.com/assets/colab-badge.svg">](https://colab.research.google.com/github/aqlaboratory/openfold/blob/main/notebooks/OpenFold.ipynb) - [paper](https://www.biorxiv.org/content/10.1101/2022.11.20.517210) 
  - a reimplementation of Alphafold2 using PyTorch

- [OmegaFold](https://github.com/HeliXonProtein/OmegaFold) - [<img src="https://colab.research.google.com/assets/colab-badge.svg">](https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/beta/omegafold.ipynb) - [paper](https://www.biorxiv.org/content/10.1101/2022.07.21.500999v1)

- [UniFold](https://github.com/dptech-corp/Uni-Fold) - [paper](https://doi.org/10.1101/2022.08.04.502811)

- [FastFold](https://github.com/hpcaitech/FastFold) - [paper](https://arxiv.org/abs/2203.00854)

- [RoseTTAFold](https://github.com/RosettaCommons/RoseTTAFold) - [<img src="https://colab.research.google.com/assets/colab-badge.svg">](https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/RoseTTAFold.ipynb) - [paper](https://www.science.org/doi/10.1126/science.abj8754)

## Multimer structure prediction

- [Alphafold2-Multimer](https://github.com/deepmind/alphafold) - [paper](https://www.biorxiv.org/content/10.1101/2021.10.04.463034v1)

- [Evolutionary Scale Modeling (ESM-2, ESMFold)](https://github.com/facebookresearch/esm) - [<img src="https://colab.research.google.com/assets/colab-badge.svg">](https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/ESMFold.ipynb) - [paper](https://www.science.org/doi/abs/10.1126/science.ade2574)

- [Uni-Fold Symmetry (UF-Symmetry)](https://github.com/dptech-corp/Uni-Fold) - [paper](https://doi.org/10.1101/2022.08.30.505833) 
  - an open-source reimplementation of Alphafold2 using PyTorch

- [AF2Complex](https://github.com/FreshAirTonight/af2complex) -  [<img src="https://colab.research.google.com/assets/colab-badge.svg">](https://colab.research.google.com/github/FreshAirTonight/af2complex/blob/main/notebook/AF2Complex_notebook.ipynb) - [paper](https://www.nature.com/articles/s41467-022-29394-2)

- [MoLPC](https://github.com/patrickbryant1/MoLPC) - [<img src="https://colab.research.google.com/assets/colab-badge.svg">](https://colab.research.google.com/github/patrickbryant1/MoLPC/blob/master/MoLPC.ipynb) - [paper](https://www.nature.com/articles/s41467-022-33729-4) 
  - an Alphafold2-based pipeline for assembling large complexes based on pairwise heterodimer prediction and Monte Carlo search.

## Design

- [ColabDesign](https://github.com/sokrypton/ColabDesign)
  - Includes notebooks for AfDesign, TrDesign, ProteinMPNN

- [ProteinMPNN](https://github.com/dauparas/ProteinMPNN) - [<img src="https://colab.research.google.com/assets/colab-badge.svg">](https://colab.research.google.com/github/sokrypton/ColabDesign/blob/v1.1.0/mpnn/examples/proteinmpnn_in_jax.ipynb) - [paper](https://www.biorxiv.org/content/10.1101/2022.06.03.494563v1)
  
- [ESM-IF1 (inverse folding)](https://github.com/facebookresearch/esm#inverse-folding-) - [paper](https://doi.org/10.1101/2022.04.10.487779)

## Peptide binding

- [AlphaFold encodes the principles to identify high affinity peptide binders (pre-print)](https://www.biorxiv.org/content/10.1101/2022.03.18.484931v1.full)

- [ColabDesign/AfDesign peptide binder design](https://github.com/sokrypton/ColabDesign/tree/main/af) - [<img src="https://colab.research.google.com/assets/colab-badge.svg">](https://colab.research.google.com/github/sokrypton/ColabDesign/blob/main/af/examples/peptide_binder_design.ipynb)

## Other lists

- [List of papers about Proteins Design using Deep Learning (Peldom)](https://github.com/Peldom/papers_for_protein_design_using_DL)
  - a *huge* well categorized list of methods, links to papers and code
  - organized by machine learning method (LSTM, CNN, GAN, VAE, Transformer etc) and mapping (Sequence -> Scaffold, Function -> Structure etc).
  - **Sections**: Benchmarks and datasets, Reviews, Model-based design, Function to Scaffold, Scaffold to Sequence, Function to Sequence, Function to Structure, Other tasks.
- [Papers on machine learning for proteins (yangkky)](https://github.com/yangkky/Machine-learning-for-proteins)
  - a big well categorized list of papers.
  - **Sections**: Reviews, Tools and datasets, Machine-learning guided directed evolution, Representation learning, Unsupervised variant prediction, Generative models, Biophysics predicting stability, Predicting structure from sequence, Predicting sequence from structure, Classification, annotation, search, and alignments, Predicting interactions with other molecules, Other supervised learning.
- [Awesome AI-based Protein Design (opendilab)](https://github.com/opendilab/awesome-AI-based-protein-design)
  - a list focusing on important peer-reviewed publications and manuscripts
- [awesome-protein-design (johnnytam100)](https://github.com/johnnytam100/awesome-protein-design)
        
## Contribution guidelines

- Should have a (theoretically) runnable implementation
  - The focus of this list is runnable software rather than pre-print/publication descriptions of implementations, but link to the pre-print/paper if you can. There are several other great lists that focus on publications.
- Prefer open source and open access 
  - When linking to publications, please preference open access versions. If the peer-reviewed publication is not open access, please link to the aRxiv/bioRxiv version when available (aRxiv/bioRxiv generally provide outgoing links to the peer-reviewed version).
- Not a historical retrospective
  - The intention is to include the best performing new implementations as they appear rather than be historically comprehensive (Andrej Sali's MODELLER was awesome, but probably obsolete at this point).
