# Stained Glass Transform (Unofficial PyTorch Implementation)

This repository contains an **unofficial PyTorch implementation** of the paper:  
[Learning Obfuscations Of LLM Embedding Sequences: Stained Glass Transform](https://arxiv.org/abs/2506.09452)  
*Jay Roberts, Kyle Mylonakis, Sidhartha Roy, and Kaan Kale*.

---

## Repository Structure

There are two main training pipelines:

- **`train_paper/`** – implementation following the paper’s description, including the MI loss.  
- **`train/`** – my own version of training, where MI is excluded for stability.

---

## Notes on MI Loss

The **Mutual Information (MI) loss** as described in the paper did not work well in my experiments:

- The **Mahalanobis term** is highly unstable in high dimensions.  
- MI values quickly explode, making optimization impractical.  

For more details, see my [training notes](training_notes.md), where I explain what worked and what didn’t.

---

## Training Recommendations

The authors are also behind [**Protopia.ai**](https://protopia.ai), which provides documentation and recommendations for training with Stained Glass Transform:

[Protopia.ai Training Recommendations for LLMs](https://docs.protopia.ai/engine/1.2.1/recommendations_for_training_llms)  


---

## Weights

Pretrained weights for two configs from `./train/config` are available on [Google Drive](https://drive.google.com/drive/folders/1tiqTURdRTvETI2ihS85AUCTPM2LDvl5c?usp=share_link).


--- 

## Experimenting Against Model Inversion Attack (MIA)

I conducted experiments to evaluate the model against MIA, since the authors did not include this analysis in their paper, despite MIA being one of the most powerful known attacks.  
A demonstration notebook is available at `./MIA/model_inversion_attack.ipynb`.
