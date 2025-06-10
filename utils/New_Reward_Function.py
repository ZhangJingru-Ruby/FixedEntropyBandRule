import torch
import math

H_MAX   = math.log(10.0)          # 10 bands → ln(10)
SIGMA_G = 0.9                     # Gini max for 10 bands

ALPHA_R = 5.0    # weight for RL reward (advantage)
ALPHA_H = 3.0    # weight for entropy-match loss
BETA_P  = 1.5    # weight for path-bias loss (near goal)

def gini_reward(probs: torch.Tensor,
                tau: torch.Tensor,     # (B,)   0=near, 1=far
                alpha=ALPHA_R):
    G       = 1.0 - (probs**2).sum(-1)           # (B,)
    G_star  = tau * SIGMA_G                      # target Gini
    R       = 1.0 - ((G - G_star) / SIGMA_G).pow(2)
    return alpha * R                             # (B,)

# ----------------------------------------------
#  2)  Custom loss: entropy-match ➕ path-bias
# ----------------------------------------------
def exploration_path_loss(probs: torch.Tensor,
                          entropy: torch.Tensor,
                          band_mean_dist: torch.Tensor,   # (B,10) 0..1
                          tau: torch.Tensor):             # (B,)
    # a) Entropy term  (match target H*)
    H_star   = tau * H_MAX
    H_star = torch.clamp(H_star, min=0.5)   # never ask for <0.5
    L_ent    = (entropy - H_star).pow(2)                  # (B,)

    # b) Path-bias term  (hug path when τ small)
    E_d      = (probs * band_mean_dist).sum(-1)           # (B,)
    w_path   = (1.0 - tau)                                # weight 0..1
    L_path   = w_path * E_d.pow(2)                        # (B,)

    return (ALPHA_H * L_ent + (BETA_P * (1.0 - tau)) * E_d.pow(2)).mean()
     # scalar


