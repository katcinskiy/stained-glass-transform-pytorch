import torch
import torch.nn as nn
import torch.nn.functional as F

class SGTLoss(nn.Module):
    def __init__(
            self, 
            embedding_weights,

            alpha_mi=0.0,
            alpha_abs_cos=1.0,
            alpha_norm=0.01
        ):
        super().__init__()
        self.embeddings_norm = embedding_weights.norm(dim=-1).detach().median()
        self.alpha_mi = alpha_mi
        self.alpha_abs_cos = alpha_abs_cos
        self.alpha_norm = alpha_norm

    def set_alpha(self, alpha_name, value):
        setattr(self, alpha_name, value)
    
    def forward(self, x, x_tilde, x_independent, mu, logvar, mu_independent, logvar_independent, 
                logits_clean, logits_obf, attention_mask):
        
        utility_loss = self._utility_loss_kl(logits_clean, logits_obf, attention_mask)
        mi_loss = self._mi_loss(x_tilde, x_independent, mu_independent, logvar, logvar_independent)
        abs_cos_loss = self._abs_cos_loss(x, x_tilde)
        norm_loss = self._median_norm_penalty(x, mu)

        scaled_mi_loss = self.alpha_mi * mi_loss
        scaled_abs_cos_loss = self.alpha_abs_cos * abs_cos_loss
        scaled_norm_loss = self.alpha_norm * norm_loss

        obfuscations_loss = scaled_mi_loss + scaled_abs_cos_loss + scaled_norm_loss

        total_loss = utility_loss + obfuscations_loss
        # total_loss = obfuscations_loss

        if total_loss < 0.05:
            self.alpha_mi = 0.00001
        
        return {
            'total_loss': total_loss,

            'obfuscations': obfuscations_loss,
            'utility': utility_loss,

            'MI': scaled_mi_loss,
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
        # logits_clean should be detached/no_grad when passed in
        log_probs_clean = F.log_softmax(logits_clean, dim=-1)
        log_probs_obf = F.log_softmax(logits_obf, dim=-1)
        
        # KL(P_obf || P_clean)
        kl_div = F.kl_div(log_probs_obf, log_probs_clean.exp(), reduction='none', log_target=False)
        
        kl_div = kl_div.sum(dim=-1) * attn_mask
        ce_loss = kl_div.sum() / attn_mask.sum()

        return ce_loss
    
    def _mi_loss(self, x_tilde, x_independent, mu_independent, logvar, logvar_independent):
        # 1. Log determinant ratio
        log_det_ratio = (logvar_independent - logvar).sum(dim=(-1, -2))
        
        # 2. Mahalanobis
        mahalanobis_distance = self._mahalanobis(x_tilde, x_independent, mu_independent, logvar_independent)
        
        # MI loss - average over batch
        return (log_det_ratio + mahalanobis_distance).mean()
    
    def _mahalanobis(self, x_tilde, x_independent, mu_independent, logvar_independent):
        vector_in_norm = (x_tilde - x_independent - mu_independent)
        logvar_independent_inverse = torch.exp(-logvar_independent)
        mahalanobis_distance = ((vector_in_norm ** 2) * logvar_independent_inverse).sum(dim=(-1, -2))
        return mahalanobis_distance

    def _median_norm_penalty(self, x, mu):
        norms = (x + mu).norm(dim=-1)
        penalty = (norms.mean() - self.embeddings_norm).abs()
        return penalty
    

def test_mi_loss_independence():
    """
    Test that MI loss approaches zero for independent random variables.
    When x_tilde and the parameters (mu_independent, logvar_independent) 
    are completely independent, the MI should be approximately zero.
    """
    print("Testing MI Loss for Independent Random Variables")
    print("=" * 50)
    
    # Create dummy embedding weights
    embedding_weights = torch.randn(1000, 512)
    loss_fn = SGTLoss(embedding_weights)
    
    # Test parameters
    batch_size = 128
    seq_len = 10
    d_model = 512
    num_trials = 5
    
    mi_values = []
    
    for trial in range(num_trials):
        print(f"\nTrial {trial + 1}/{num_trials}")
        
        # Generate completely independent random variables
        # x_tilde: random obfuscated embeddings
        x_tilde = torch.randn(batch_size, seq_len, d_model)
        
        # x_independent: independent random embeddings (not used in MI calc but for consistency)
        x_independent = torch.randn(batch_size, seq_len, d_model)
        
        # mu_independent: independent random means
        mu_independent = torch.randn(batch_size, seq_len, d_model)
        
        # logvar and logvar_independent: independent random log variances
        logvar = torch.randn(batch_size, seq_len, d_model)
        logvar_independent = torch.randn(batch_size, seq_len, d_model)
        
        # Compute MI loss
        mi_loss = loss_fn._mi_loss(x_tilde, x_independent, mu_independent, logvar, logvar_independent)
        mi_values.append(mi_loss.item())
        
        print(f"  MI Loss: {mi_loss.item():.6f}")
    
    avg_mi = sum(mi_values) / len(mi_values)
    std_mi = torch.tensor(mi_values).std().item()
    
    print(f"\nResults Summary:")
    print(f"Average MI Loss: {avg_mi:.6f}")
    print(f"Standard Deviation: {std_mi:.6f}")
    print(f"Range: [{min(mi_values):.6f}, {max(mi_values):.6f}]")
    
    # For truly independent variables, MI should be close to zero
    # But due to finite sampling, we expect some variation
    if abs(avg_mi) < 1.0:  # Reasonable threshold for "close to zero"
        print("✅ PASS: MI loss is close to zero for independent variables")
    else:
        print("❌ FAIL: MI loss is too large for independent variables")
    
    return mi_values


test_mi_loss_independence()