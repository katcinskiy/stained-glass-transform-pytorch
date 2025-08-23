import torch
import torch.nn as nn
import torch.nn.functional as F

class SGTLossPractical(nn.Module):
    def __init__(
            self, 
            embedding_weights,

            alpha_utility,
            alpha_obfuscation,
            alpha_abs_cos,
            alpha_log_det_ratio

        ):
        super().__init__()
        self.embeddings_norm = embedding_weights.norm(dim=-1).detach().median()

        self.alpha_utility = alpha_utility
        self.alpha_obfuscation = alpha_obfuscation
        self.alpha_abs_cos = alpha_abs_cos
        self.alpha_log_det_ratio = alpha_log_det_ratio
    
    def forward(self, x, x_tilde, x_independent, mu, logvar, mu_independent, logvar_independent, 
                logits_clean, logits_obf, attention_mask, independent_attention_mask):
        
        utility_loss = self._utility_loss_kl(logits_clean, logits_obf, attention_mask)
        log_det_ratio_loss = self._log_det_ratio_loss(logvar, logvar_independent, attention_mask & independent_attention_mask)
        abs_cos_loss = self._abs_cos_loss(x, x_tilde)
        # norm_loss = self._median_norm_penalty(x, mu)

        scaled_log_det_ratio_loss = self.alpha_log_det_ratio * log_det_ratio_loss
        
        scaled_abs_cos_loss = self.alpha_abs_cos * abs_cos_loss

        obfuscations_loss = scaled_log_det_ratio_loss + scaled_abs_cos_loss

        total_loss = self.alpha_utility * utility_loss + self.alpha_obfuscation * obfuscations_loss
        
        return {
            'total_loss': total_loss,

            'obfuscations': obfuscations_loss,
            'utility': utility_loss,

            'log_det_ratio': scaled_log_det_ratio_loss,
            'abs_cos': scaled_abs_cos_loss,

            'raw/abs_cos': abs_cos_loss
        }

    def _abs_cos_loss(self, x, x_tilde):
        cos_sim = F.cosine_similarity(x, x_tilde, dim=-1)
        return cos_sim.abs().mean()
    
    def _utility_loss_kl(self, logits_clean, logits_obf, attn_mask):
        # logits_clean should be detached/no_grad when passed in
        log_probs_clean = F.log_softmax(logits_clean, dim=-1)
        log_probs_obf = F.log_softmax(logits_obf, dim=-1)
        
        kl_div = F.kl_div(log_probs_obf, log_probs_clean.exp(), reduction='none', log_target=False)
        
        kl_div = kl_div.sum(dim=-1) * attn_mask
        ce_loss = kl_div.sum() / attn_mask.sum()

        return ce_loss

    def _log_det_ratio_loss(self, logvar, logvar_independent, attention_mask):
        # mask = attention_mask.unsqueeze(-1).float()
        # denom = mask.sum(dim=(-1, -2)).clamp_min(1.0)
        # log_det_ratio = ((logvar_independent.detach() - logvar) * mask).sum(dim=(-1, -2))
        # return log_det_ratio.mean()
    
        mask = attention_mask.unsqueeze(-1).float()
        denom = mask.sum(dim=(-1, -2)).clamp_min(1.0)
        log_det_ratio = ((logvar ** 2) * mask)
        return log_det_ratio.mean()


    def _median_norm_penalty(self, x, mu):
        norms = (x + mu).norm(dim=-1)
        penalty = (norms.mean() - self.embeddings_norm).abs()
        return penalty
