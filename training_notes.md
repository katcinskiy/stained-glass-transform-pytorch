# Training Notes on SGT

### Intuition behind each Loss component

- $\mathcal{L}_O^{\mathrm{MNP}}$ constrains the obfuscation so that the resulting embeddings remain within the distribution of original embeddings in terms of norm.  
- $\mathcal{L}_O^{\mathrm{ACS}}$ encourages clean and obfuscated embeddings to be orthogonal.  
- $\mathcal{L}_O^{\mathrm{MI}}$ aims to reduce the mutual information between obfuscated and clean embeddings.

The utility loss is more straightforward, as it typically relies on KL-divergence or cross-entropy for knowledge distillation.

---

## My Training Notes

1. The MI component proved unstable. The Mahalanobis distance term produced extremely large values (on the order of ~50,000), and scaling $\alpha_1$ did not resolve the issue.  

2. There are discrepancies between the official website and the paper. In particular, Protopia’s [documentation](https://docs.protopia.ai/engine/1.2.1/recommendations_for_training_llms) only refers to the `std_log_ratio_loss_weight` parameter, which suggests that the Mahalanobis term may have been removed. However, relying solely on the log-det ratio also seems problematic: near `i` covariance the gradient is nearly constant, which tends to push log-variances to infinity.  

3. As an alternative, I experimented with an MSE penalty on logvar, driving it toward 0 so that the standard deviation converges to 1. While this might theoretically strengthen obfuscation, in practice it severely degraded utility.  

4. I eventually omitted both the MI and MNP components. A simplified loss using only KL + cosine similarity produced stable training and acceptable utility performance but it gives no theoretical guarantees of privacy.

5. It is better to train on large dataset even if you just testing to reduce overfitting
