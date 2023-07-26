# Awesome Protein Structure Prediction and Design Software List

A collection of software for protein structure prediction and design, with a focus on new deep learning and transformer based tools.

## Table of Contents

- [Structure prediction](#structure-prediction)
- [Multimer structure prediction](#multimer-structure-prediction)
- [Design](#design)
- [Peptide binding](#peptide-binding)
- [General protein language models](#general-protein-language-models)
- [Other lists](#other-lists)
- [Uncurated searches](#uncurated-searches)
- [Contribution guidelines](#contribution-guidelines)

## Structure prediction

- [Alphafold2](https://github.com/deepmind/alphafold) - [<img src="https://colab.research.google.com/assets/colab-badge.svg">](https://colab.research.google.com/github/deepmind/alphafold/blob/main/notebooks/AlphaFold.ipynb) - [paper](https://doi.org/10.1038/s41586-021-03819-2)

- [ColabFold](https://github.com/sokrypton/ColabFold) - [Colab notebooks](https://github.com/sokrypton/ColabFold#making-protein-folding-accessible-to-all-via-google-colab) - [paper](https://doi.org/10.5281/zenodo.5123296) 
  - a collection of community-developed Colab notebooks with extra features for Alphafold2, ESMFold, RosettaFold, OmegaFold in various modes.

- [ParaFold](https://github.com/Zuricho/ParallelFold) - [paper](https://arxiv.org/abs/2111.06340) 
  - a lightly modified fork of Alphafold2 which splits the pipeline into seperate MSA generation and GPU inference steps for better use of computing resources.

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

- [PeSTo](https://github.com/LBM-EPFL/PeSTo) - [paper](https://www.nature.com/articles/s41467-023-37701-8) - [<img src="https://colab.research.google.com/assets/colab-badge.svg">](https://colab.research.google.com/github/LBM-EPFL/PeSTo/blob/main/apply_model.ipynb) - [web app](https://pesto.epfl.ch/)
  - Predicts interface residues from structure, for protein-protein, protein-DNA/RNA and protein-ligand interfaces.

- [TCRdock](https://github.com/phbradley/TCRdock) - - [paper](https://elifesciences.org/articles/82813)
  - A T cell receptor:peptide-MHC docking protocol using an Alphafold model finetuned for TCR:peptide+MHC complexes.

## Design

- [ColabDesign](https://github.com/sokrypton/ColabDesign)
  - Includes notebooks for AfDesign, TrDesign, ProteinMPNN

- [ProteinMPNN](https://github.com/dauparas/ProteinMPNN) - [<img src="https://colab.research.google.com/assets/colab-badge.svg">](https://colab.research.google.com/github/sokrypton/ColabDesign/blob/v1.1.0/mpnn/examples/proteinmpnn_in_jax.ipynb) - [paper](https://www.biorxiv.org/content/10.1101/2022.06.03.494563v1)

- ESM methods
  - [ESM-IF1 (inverse folding)](https://github.com/facebookresearch/esm#inverse-folding-) - [paper](https://doi.org/10.1101/2022.04.10.487779)
  - [ESMFold-based constraint based design via "Protein programming language"](https://github.com/facebookresearch/esm/tree/main/examples/protein-programming-language) - [paper](https://www.biorxiv.org/content/10.1101/2022.12.21.521526v1.full) -  [<img src="https://colab.research.google.com/assets/colab-badge.svg">](https://colab.research.google.com/github/facebookresearch/esm/blob/main/examples/protein-programming-language/tutorial.ipynb)
    - A "high-level programming language for generative protein design". Hopefully this method is given a catchier name for the peer-reviewed publication.
  - [ESM-2 language model design](https://github.com/facebookresearch/esm/tree/main/examples/lm-design) - [paper](https://www.biorxiv.org/content/10.1101/2022.12.21.521521v1.full) - [<img src="https://colab.research.google.com/assets/colab-badge.svg">](https://colab.research.google.com/github/facebookresearch/esm/blob/main/examples/lm-design/free_generation.ipynb) | [<img src="https://colab.research.google.com/assets/colab-badge.svg">](https://colab.research.google.com/github/facebookresearch/esm/blob/main/examples/lm-design/fixed_backbone.ipynb)
    - Fixed-backbone and free generative design, apparently capable of generalizing to produce sequences of folded proteins with no detectable sequence homoology with natural proteins.
 
- [ECNet](https://github.com/luoyunan/ECNet) - [paper](https://www.nature.com/articles/s41467-021-25976-8)
  - Fine-tunable model that predicts protein fitness/function from sequence. Can be used to prioritize variants when optimizing function based on existing data. 

## Peptide binding

- [AlphaFold encodes the principles to identify high affinity peptide binders (pre-print)](https://www.biorxiv.org/content/10.1101/2022.03.18.484931v1.full)

- [ColabDesign/AfDesign peptide binder design](https://github.com/sokrypton/ColabDesign/tree/main/af) - [<img src="https://colab.research.google.com/assets/colab-badge.svg">](https://colab.research.google.com/github/sokrypton/ColabDesign/blob/main/af/examples/peptide_binder_design.ipynb)

- [Solubility aware protein-binding peptide design with AfDesign](https://github.com/ohuelab/Solubility_AfDesign) - [<img src="https://colab.research.google.com/assets/colab-badge.svg">](https://colab.research.google.com/github/ohuelab/Solubility_AfDesign/blob/solubility/design.ipynb) - [paper](https://www.mdpi.com/2227-9059/10/7/1626)
  - Based on ColabDesign/AfDesign, with an extra solubility objective function

## Sequence generation

- [ProtGPT2](https://huggingface.co/nferruz/ProtGPT2) - [paper](https://www.nature.com/articles/s41467-022-32007-7)
  - A generative transformer model based on GPT-2, at generates native-like sequences

## General protein language models

- [ProtTrans](https://github.com/agemagician/ProtTrans) - [paper](https://ieeexplore.ieee.org/document/9477085) - a transformer model of protein sequence (ProtT5)
  - embeddings that are competative with ESM-1b on subcellular localization prediction
  - [structure prediction using EMBER2 and trRosetta](https://github.com/kWeissenow/EMBER2) - lower resource but can't match Alphafold2

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

## Uncurated searches

- [Github repos tagged protein-design](https://github.com/topics/protein-design)
  - results here may find their way into the curated list above .... 

## Contribution guidelines

- Should have a (theoretically) runnable implementation
  - The focus of this list is runnable software rather than pre-print/publication descriptions of implementations, but link to the pre-print/paper if you can. There are several other great lists that focus on publications.
- Prefer open source and open access 
  - When linking to publications, please preference open access versions. If the peer-reviewed publication is not open access, please link to the aRxiv/bioRxiv version when available (aRxiv/bioRxiv generally provide outgoing links to the peer-reviewed version).
- Not a historical retrospective
  - The intention is to include the best performing new implementations as they appear rather than be historically comprehensive (Andrej Sali's MODELLER was awesome, but probably obsolete at this point).
