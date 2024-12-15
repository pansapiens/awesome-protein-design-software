# Awesome Protein Structure Prediction and Design Software List

A collection of software for protein structure prediction and design, with a focus on new deep learning and transformer based tools.

## Table of Contents

- [Structure prediction](#structure-prediction)
- [Multimer structure prediction](#multimer-structure-prediction)
- [Design](#design)
- [Peptide and ligand binding](#peptide-and-ligand-binding)
- [Sequence generation](#sequence-generation)
- [General protein language models](#general-protein-language-models)
- [Tutorials and Workshops](#tutorials-and-workshops)
- [Other lists](#other-lists)
- [Uncurated searches](#uncurated-searches)
- [Contribution guidelines](#contribution-guidelines)

## Structure prediction

- [Alphafold3](https://github.com/google-deepmind/alphafold3) - [paper](https://doi.org/10.1038/s41586-024-07487-w) - [blog](https://blog.google/technology/ai/google-deepmind-isomorphic-alphafold-3-ai-model/) - [web app](https://alphafoldserver.com/)
  - an improved Alphafold with support for nucleotide, ligand and post-translational modification modelling
  - ⚠️ _be aware of the restrictive license associated with the model weights_

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

- [ManyFold](https://github.com/instadeepai/manyfold) - [paper](https://doi.org/10.1093/bioinformatics/btac773)
  - a framework for batch validation and distributed training with Alphafold, Openfold, others.
  - includes pLMFold, a ESM-1b-based structure prediction model

- [Boltz-1](https://github.com/jwohlwend/boltz) - [paper](https://doi.org/10.1101/2024.11.19.624167)
  - structure prediction of proteins, RNA, DNA, and small molecules, modified residues, covalent ligands and glycans
  - conditional generation of pocket residues

- [Chai-1](https://github.com/chaidiscovery/chai-lab) - [paper](https://www.biorxiv.org/content/10.1101/2024.10.10.615955) - [server](https://lab.chaidiscovery.com/)
  - enables unified prediction of proteins, small molecules, DNA, RNA, glycosylations, including experimental restraints.
 
- [HelixFold3](https://github.com/PaddlePaddle/PaddleHelix/tree/dev/apps/protein_folding/helixfold3) - [paper](https://arxiv.org/pdf/2408.16975)
  - structure prediction aiming to replicate Alphafold3 (⚠️ _non-commercial use only_)
  - part of the [PaddleHelix](https://github.com/PaddlePaddle/PaddleHelix) suite for structure prediction and design for protein, DNA, RNA and small molecules.


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

- [TCRdock](https://github.com/phbradley/TCRdock) - [paper](https://elifesciences.org/articles/82813)
  - A T cell receptor:peptide-MHC docking protocol using an Alphafold model finetuned for TCR:peptide+MHC complexes.
 
- [AlphaPulldown](https://github.com/KosinskiLab/AlphaPulldown) - [paper](https://doi.org/10.1093/bioinformatics/btac749) - [website](https://www.embl-hamburg.de/AlphaPulldown/)
  - computational protein-protein interaction screening using AlphaFold Multimer

- [Boltz-1](https://github.com/jwohlwend/boltz)
  -  predicts the 3D structure of proteins, RNA, DNA, small molecules, modified residues, covalent ligands and glycans, conditional generation of pockets  

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

- [ProteinSolver](https://github.com/ostrokach/proteinsolver) - [paper](https://doi.org/10.1016/j.cels.2020.08.016) -  [<img src="https://colab.research.google.com/assets/colab-badge.svg">](https://colab.research.google.com/github/ostrokach/proteinsolver/blob/master/notebooks/20_protein_demo.ipynb)
  - a graph neural network for constraint based protein structure solving, aiding design of proteins that fold into a predetermined geometric shape.
 
- [RFDiffusion](https://github.com/RosettaCommons/RFdiffusion) - [paper](https://www.biorxiv.org/content/10.1101/2022.12.09.519842v1) - [<img src="https://colab.research.google.com/assets/colab-badge.svg">](https://colab.research.google.com/github/sokrypton/ColabDesign/blob/v1.1.1/rf/examples/diffusion.ipynb)
  - a generative diffusion model for protein (backbone) structure generation, with or without conditional information (eg motif or binding target) 

- [LM-Design](https://github.com/BytedProtein/ByProt) and ByProt - [paper](https://arxiv.org/abs/2302.01649)

- [InstructPLM](https://github.com/Eikor/InstructPLM) - [paper](https://doi.org/10.1101/2024.04.17.589642)

- [BindCraft](https://github.com/martinpacesa/BindCraft) - [paper](https://doi.org/10.1101/2024.09.30.615802) - [<img src="https://colab.research.google.com/assets/colab-badge.svg">](https://colab.research.google.com/github/martinpacesa/BindCraft/blob/main/notebooks/BindCraft.ipynb)

## Peptide and ligand binding

- [AlphaFold encodes the principles to identify high affinity peptide binders (pre-print)](https://www.biorxiv.org/content/10.1101/2022.03.18.484931v1.full)

- [ColabDesign/AfDesign peptide binder design](https://github.com/sokrypton/ColabDesign/tree/main/af) - [<img src="https://colab.research.google.com/assets/colab-badge.svg">](https://colab.research.google.com/github/sokrypton/ColabDesign/blob/main/af/examples/peptide_binder_design.ipynb)

- [Solubility aware protein-binding peptide design with AfDesign](https://github.com/ohuelab/Solubility_AfDesign) - [<img src="https://colab.research.google.com/assets/colab-badge.svg">](https://colab.research.google.com/github/ohuelab/Solubility_AfDesign/blob/solubility/design.ipynb) - [paper](https://www.mdpi.com/2227-9059/10/7/1626)
  - Based on ColabDesign/AfDesign, with an extra solubility objective function

- [DiffBindFR](https://github.com/HBioquant/DiffBindFR)
  - diffusion model based flexible protein-ligand docking
 
- [AlphaFill](https://github.com/PDB-REDO/alphafill) - [web app](https://alphafill.eu/) - [paper](https://www.nature.com/articles/s41592-022-01685-y)
  - "transplants" missing ligands, cofactors and (metal) ions to the AlphaFold models.

- [PepMLM](https://github.com/programmablebio/pepmlm) - [<img src="https://colab.research.google.com/assets/colab-badge.svg">](https://colab.research.google.com/drive/1u0i-LBog_lvQ5YRKs7QLKh_RtI-tV8qM?usp=sharing) - [paper](https://doi.org/10.48550/arXiv.2310.03842)
  - A linear peptide binder sequence generation model, using ESM-2.
  - A nice tutorial example using [PepMLM and EvoProtGrad](https://huggingface.co/blog/AmelieSchreiber/esm-interact) for linear peptide binding design.

## Sequence generation

- [ProtGPT2](https://huggingface.co/nferruz/ProtGPT2) - [paper](https://www.nature.com/articles/s41467-022-32007-7)
  - A generative transformer model based on GPT-2, at generates native-like sequences
- [EvoDiff](https://github.com/microsoft/evodiff) - [paper](https://www.biorxiv.org/content/10.1101/2023.09.11.556673v1)
  - Generation of protein sequences and evolutionary alignments via discrete diffusion models. Also explores generation in intrinsically disordered regions (IDRs). 
- [PoET](https://github.com/OpenProteinAI/PoET) - [paper](https://doi.org/10.48550/arXiv.2306.06156)
  - protein language model for variant effect prediction and conditional sequence generation.
- [EvoProtGrad](https://github.com/NREL/EvoProtGrad) - [paper](https://doi.org/10.1088/2632-2153/accacd)
  - in-silico directed evolution of sequences with MCMC sampling and gradients from supervised models

## Sequence similarity search and (structural) alignment

- [DeepBLAST](https://github.com/flatironinstitute/deepblast) - [paper](https://doi.org/10.1101/2020.11.03.365932)
  - Pairwise alignments that better reflect structural alignment, using protein language model embeddings and differentiable dynamic programming for Smith-Waterman or Needleman-Wunch alignment.
  - Associated repo: [TM-vec](https://github.com/tymor22/tm-vec) 

## General protein language models

- [ProtTrans](https://github.com/agemagician/ProtTrans) - [paper](https://ieeexplore.ieee.org/document/9477085) - a transformer model of protein sequence (ProtT5)
  - embeddings that are competative with ESM-1b on subcellular localization prediction
  - [structure prediction using EMBER2 and trRosetta](https://github.com/kWeissenow/EMBER2) - lower resource but can't match Alphafold2
 
- [AMPLIFY](https://github.com/chandar-lab/AMPLIFY) - [paper](https://doi.org/10.1101/2024.09.23.614603) - [<img src="https://colab.research.google.com/assets/colab-badge.svg">](https://colab.research.google.com/github/chandar-lab/AMPLIFY/blob/rc-0.1/examples/)
  - protein language models reimplementing ESM2 with improved inferance and training performance, with open data + pre-training code
  - demonstrates the impact of data set size and curation on protein language model performance

## Tutorials and workshops

- [DL4Proteins workshops](https://github.com/Graylab/DL4Proteins-notebooks) - [<img src="https://colab.research.google.com/assets/colab-badge.svg">](https://colab.research.google.com/github/Graylab/DL4Proteins-notebooks/)
  - instructive notebooks covering the basics of neural networks and PyTorch, through graph neural networks, Denoising Diffusion Probabilistic Models, Alphafold2 and RFDiffusion

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
- [Scientific Large Language Models (Sci-LLMs)](https://github.com/HICAI-ZJU/Scientific-LLM-Survey?tab=readme-ov-file#protein-sequence-generationdesign)
  - a fantastic list of miletone papers in scientific large language models, with a "Protein Sequence Generation/Design" section
- [awesome-protein-design (johnnytam100)](https://github.com/johnnytam100/awesome-protein-design)
- [awesome-protein-representation-learning](https://github.com/LirongWu/awesome-protein-representation-learning)
- [folding_tools](https://github.com/duerrsimon/folding_tools) - folded prediction tools list

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
