import torch
import torch.nn as nn

class SGTModel(nn.Module):
    def __init__(self, d, nhead, ff, layers, mu_init_weight, mu_init_bias, logvar_init_weight, logvar_init_bias):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(d_model=d, nhead=nhead, 
                                             dim_feedforward=ff*d, batch_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.mu_head = nn.Linear(d, d)
        self.logvar_head = nn.Linear(d, d)

        self._initilize_head_weights(mu_init_weight, mu_init_bias, logvar_init_weight, logvar_init_bias)
        
    def forward(self, x, attention_mask):
        hidden_embeds = self.enc(x, src_key_padding_mask=~attention_mask.bool())
        mu = self.mu_head(hidden_embeds)
        logvar = self.logvar_head(hidden_embeds)
        return mu, logvar
    
    def sample(self, x, attention_mask):
        mu, logvar = self(x, attention_mask)
        logvar = torch.clamp(logvar, min=-10, max=2)
        eps = torch.randn_like(mu)
        z = x + mu + eps * torch.exp(0.5 * logvar)
        return z, mu, logvar
    
    def _initialize_tensor(self, tensor, val):
        if val == 0:
            nn.init.zeros_(tensor)
        else:
            nn.init.constant_(tensor, val)

    def _initilize_head_weights(self, mu_init_weight, mu_init_bias, logvar_init_weight, logvar_init_bias):
        if mu_init_weight is not None:
            print(f"Initializing mu_head.weight with: {mu_init_weight}")
            self._initialize_tensor(self.mu_head.weight, mu_init_weight)
        else:
            print("mu_init_weight is None - skipping mu_head.weight initialization")

        if mu_init_bias is not None:
            print(f"Initializing mu_head.bias with: {mu_init_bias}")
            self._initialize_tensor(self.mu_head.bias, mu_init_bias)
        else:
            print("mu_init_bias is None - skipping mu_head.bias initialization")

        if logvar_init_weight is not None:
            print(f"Initializing logvar_head.weight with: {logvar_init_weight}")
            self._initialize_tensor(self.logvar_head.weight, logvar_init_weight)
        else:
            print("logvar_init_weight is None - skipping logvar_head.weight initialization")

        if logvar_init_bias is not None:
            print(f"Initializing logvar_head.bias with: {logvar_init_bias}")
            self._initialize_tensor(self.logvar_head.bias, logvar_init_bias)
        else:
            print("logvar_init_bias is None - skipping logvar_head.bias initialization")
