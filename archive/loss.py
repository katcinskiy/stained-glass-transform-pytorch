import torch
import torch.nn as nn
import torch.nn.functional as F

class SGTLoss(nn.Module):
    def __init__(
            self, 
            embedding_weights,
            alpha_mi,
            alpha_cos,
            alpha_norm,
            alpha_utility,
            alpha_obfuscation
        ):
        super().__init__()
        self.embeddings_norm = embedding_weights.norm(dim=-1).detach().median()
        self.alpha_mi = alpha_mi
        self.alpha_cos = alpha_cos
        self.alpha_norm = alpha_norm
        self.alpha_utility = alpha_utility
        self.alpha_obfuscation = alpha_obfuscation

    def set_alpha(self, alpha_name, value):
        setattr(self, alpha_name, value)
    
    def forward(self, x, x_independent, llm, sgt, attention_mask):
        x_tilde, mu, logvar = sgt.sample(x)

        utility_loss = self._utility_loss_kl(llm, x, x_tilde, attention_mask)

        mi_loss = self._mi_loss(x, x_independent, x_tilde, logvar, sgt)
        abs_cos_loss = self._abs_cos_loss(x, x_tilde)
        norm_loss = self._median_norm_penalty(x, mu)

        scaled_utility_loss = self.alpha_utility * utility_loss
        
        scaled_mi_loss = self.alpha_mi * mi_loss
        scaled_cos_loss = self.alpha_cos * abs_cos_loss
        scaled_norm_loss = self.alpha_norm * norm_loss

        obfuscations_loss = scaled_mi_loss + scaled_cos_loss + scaled_norm_loss
        total_loss = scaled_utility_loss + self.alpha_obfuscation * obfuscations_loss

        return {
            'total_loss': total_loss,
            'obfuscations_loss': obfuscations_loss,
            'utility_loss': utility_loss,
            'mi_loss': mi_loss,
            'abs_cos_loss': abs_cos_loss,
            'norm_loss': norm_loss,
            'scaled_utility_loss': scaled_utility_loss,
            'scaled_mi_loss': scaled_mi_loss,
            'scaled_cos_loss': scaled_cos_loss,
            'scaled_norm_loss': scaled_norm_loss
        }


    def _abs_cos_loss(self, x, x_tilde):
        cos_sim = F.cosine_similarity(x, x_tilde, dim=-1) # shape (b, l, d)
        return cos_sim.abs().mean()
    
    def _utility_loss_ce(self, llm, x, x_tilde, attn_mask):
        with torch.no_grad():
            logits_clean = llm(inputs_embeds=x, attention_mask=attn_mask).logits
        
        logits_obf = llm(inputs_embeds=x_tilde, attention_mask=attn_mask).logits

        x_probas = F.softmax(logits_clean, dim=-1)
        
        x_tilde_log_probas = F.log_softmax(logits_obf, dim=-1)
        ce_loss = (-x_probas * x_tilde_log_probas).sum(dim=-1)

        ce_loss = ce_loss * attn_mask
        ce_loss = ce_loss.sum() / attn_mask.sum() 

        return ce_loss
    
    def _utility_loss_kl(self, llm, x, x_tilde, attn_mask):
        with torch.no_grad():
            logits_clean = llm(inputs_embeds=x, attention_mask=attn_mask).logits
        
        logits_obf = llm(inputs_embeds=x_tilde, attention_mask=attn_mask).logits

        # CHANGE THIS: Use KL divergence instead of your current approach
        log_probs_clean = F.log_softmax(logits_clean, dim=-1).detach()
        log_probs_obf = F.log_softmax(logits_obf, dim=-1)
        
        # KL(P_obf || P_clean)
        kl_div = F.kl_div(log_probs_obf, log_probs_clean.exp(), reduction='none', log_target=False)
        
        kl_div = kl_div.sum(dim=-1) * attn_mask
        ce_loss = kl_div.sum() / attn_mask.sum()

        return ce_loss
    
    def _mi_loss(self, x, x_independent, x_tilde, logvar, sgt):

        mu_independent, logvar_independent = sgt(x_independent)
        
        # 1. Log determinant ratio
        log_det_ratio = (logvar_independent - logvar).sum(dim=(-1, -2))
        
        # 2. Mahalanobis
        mahalanobis_distance = self._mahalanobis(x_tilde, x_independent, mu_independent, logvar_independent)
        
        # MI loss - среднее по батчу
        return (log_det_ratio + mahalanobis_distance).mean()
    
    def _mahalanobis(self, x_tilde, x_independent, mu_independent, logvar_independent):
        # TODO: test it
        vector_in_norm = (x_tilde - x_independent - mu_independent)

        logvar_independent_inverse = torch.exp(-logvar_independent)
        
        mahalanobis_distance = ((vector_in_norm ** 2) * logvar_independent_inverse).sum(dim=(-1, -2))

        return mahalanobis_distance

    def _median_norm_penalty(self, x, mu):
        norms = (x + mu).norm(dim=-1)
        
        penalty = (norms.mean() - self.embeddings_norm).abs()

        return penalty