# Logit Lens Meets Vision Transformers: Layer Analysis in DINO

This repository explores how the Logit Lens technique — originally proposed for analyzing language models — can be adapted and applied to self-supervised vision transformers, specifically the DINO model. The main focus is to investigate how intermediate representations evolve across layers and to identify redundant or less informative layers using various similarity metrics and probing methods.

---

## 📁 Notebooks Overview

#### `01_logit_lens_introduction.ipynb`
A detailed introduction to the Logit Lens method. Includes explanatory comments and pseudocode inspired by the blog post:  
[Interpreting GPT: the Logit Lens](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens)

#### `02_logit_lens_reproduction.ipynb`
A faithful reproduction of the original Logit Lens implementation provided in the above blog post. This serves as a reference baseline for understanding the method before adapting it to visual models.

#### `03_logit_lens_vit_analysis.ipynb`
First experimental application of Logit Lens to a Vision Transformer. Here, we use a DINO-trained ViT and compute cosine similarity between each intermediate layer and the final output representation to observe how the model's understanding develops.

#### `04_layer_skipping_analysis_DINO.ipynb`
Comprehensive layer analysis aiming to detect and justify the removal of potentially redundant layers in DINO. This is done using a combination of cosine similarity, CKA (Canonical Correlation Analysis), and CKNNA (Centered Kernel Normalized Nearest Neighbors) to measure inter-layer representation similarity.

#### `05_analysis_of_layer_skipping_dino_with_linear_probing.ipynb`
This notebook evaluates how skipping certain layers affects downstream performance. Linear classifiers are trained on frozen representations from different configurations to assess the impact of layer removal on accuracy.

#### `06_greedy_layer_selection_analysis.ipynb`
This notebook analyzes the contribution of individual transformer layers in DINO ViT-S16 and ViT-B8 models using a greedy selection strategy. The goal is to identify the smallest subset of layers needed to reach a target accuracy (e.g., 0.9), and compare the performance and layer selection patterns between the two models.

