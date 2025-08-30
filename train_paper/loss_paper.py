import torch
import torch.nn as nn
import torch.nn.functional as F

class SGTLossPaper(nn.Module):
    def __init__(
            self, 
            embedding_weights,
            alpha_mi=0.0,
            alpha_abs_cos=0.0,
            alpha_norm=0.0,
        ):
        super().__init__()
        self.embeddings_norm = embedding_weights.norm(dim=-1).detach().median()

        self.alpha_mi = alpha_mi
        self.alpha_abs_cos = alpha_abs_cos
        self.alpha_norm = alpha_norm


    def set_alpha(self, alpha_name, value):
        setattr(self, alpha_name, value)
    
    def forward(self, x, x_tilde, x_independent, mu, logvar, mu_independent, logvar_independent, 
                logits_clean, logits_obf, attention_mask, independent_attention_mask):
        
        utility_loss = self._utility_loss_kl(logits_clean, logits_obf, attention_mask)
        mi_loss, log_det_ratio, mahalanobis = self._mi_loss(x_tilde, x_independent, mu_independent, logvar, logvar_independent, attention_mask & independent_attention_mask)
        abs_cos_loss = self._abs_cos_loss(x, x_tilde)
        norm_loss = self._median_norm_penalty(x, mu)

        scaled_mi_loss = self.alpha_mi * mi_loss
        scaled_log_det_ratio = self.alpha_mi * log_det_ratio
        scaled_mahalanobis = self.alpha_mi * mahalanobis
        scaled_abs_cos_loss = self.alpha_abs_cos * abs_cos_loss
        scaled_norm_loss = self.alpha_norm * norm_loss

        obfuscations_loss = scaled_mi_loss + scaled_abs_cos_loss + scaled_norm_loss

        total_loss = utility_loss + obfuscations_loss
        
        return {
            'total_loss': total_loss,

            'obfuscations': obfuscations_loss,
            'utility': utility_loss,

            'MI': scaled_mi_loss,
            'log_det_ratio': scaled_log_det_ratio,
            'mahalanobis': scaled_mahalanobis,
            'abs_cos': scaled_abs_cos_loss,
            'norm': scaled_norm_loss,

            'raw/MI': mi_loss,
            'raw/abs_cos': abs_cos_loss,
            'raw/norm': scaled_norm_loss,

            'alphas/MI': torch.tensor(self.alpha_mi),
            'alphas/abs_cos': torch.tensor(self.alpha_abs_cos),
            'alphas/norm': torch.tensor(self.alpha_norm)
        }

    def _abs_cos_loss(self, x, x_tilde):
        cos_sim = F.cosine_similarity(x, x_tilde, dim=-1)
        return cos_sim.abs().mean()
    
    def _utility_loss_kl(self, logits_clean, logits_obf, attn_mask):
        # logits_clean should be detached/no_grad when passed
        log_probs_clean = F.log_softmax(logits_clean, dim=-1)
        log_probs_obf = F.log_softmax(logits_obf, dim=-1)
        
        kl_div = F.kl_div(log_probs_obf, log_probs_clean.exp(), reduction='none', log_target=False)
        
        kl_div = kl_div.sum(dim=-1) * attn_mask
        ce_loss = kl_div.sum() / attn_mask.sum()

        return ce_loss

    def _mi_loss(self, x_tilde, x_independent, mu_independent,
             logvar, logvar_independent, attention_mask):
        
        mask = attention_mask.unsqueeze(-1).float()

        denom = mask.sum(dim=(-1, -2)).clamp_min(1.0)

        log_det_ratio = ((logvar_independent - logvar) * mask).sum(dim=(-1, -2))

        vec = x_tilde - x_independent - mu_independent
        inv = torch.exp(-logvar_independent)

        # this mahalanobis guy is SUPER unstable, no way we can train it. It has values around 50000 for seq length 32
        mahalanobis = ((vec.pow(2) * inv) * mask).sum(dim=(-1, -2))

        return (log_det_ratio + mahalanobis).mean(), log_det_ratio.mean(), mahalanobis.mean()

    def _median_norm_penalty(self, x, mu):
        norms = (x + mu).norm(dim=-1)
        penalty = (norms.mean() - self.embeddings_norm).abs()
        return penalty
