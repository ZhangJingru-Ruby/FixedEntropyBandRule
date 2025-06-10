"""
Rule-based sub-goal band sampler with fixed-entropy
    τ  (0 = near goal, 1 = far)  →  probability over 10 bands
    
"""

import os, csv, time, torch
from torch.utils.data import DataLoader, random_split

from utils.data_loader          import load_data_for_environment, SubgoalDataset, custom_collate_fn
from utils.New_Reward_Function  import gini_reward          # we still log RL reward for comparison
from utils.environment_functions import visualize_band_probs

# ------------------------------------------------------------------
#  RULE-BASED PROBABILITY FUNCTION
# ------------------------------------------------------------------
def rule_probs_temp(tau, N=10, T_min=0.05, T_max=6.0):
    idx = torch.arange(N, device=tau.device)
    T   = T_min + tau * (T_max - T_min)      # hotter as τ ↑
    logits = -idx / T.unsqueeze(-1)          # (B,N)
    probs  = torch.softmax(logits, dim=-1)
    return probs

# ------------------------------------------------------------------
#  TRAIN / EVAL LOOP (no network, no gradients)
# ------------------------------------------------------------------
def main():

    BATCH_SIZE = 32
    NUM_EPOCHS_PER_ENV = 50
    ALPHA_R    = 5.0           # keep reward scale for logging

    run_id  = f"rule_{time.strftime('%m%d_%H%M')}"
    run_dir = os.path.join("results", run_id)
    os.makedirs(run_dir, exist_ok=True)

    csv_path = os.path.join(run_dir, f"log_{run_id}.csv")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["env", "epoch", "val_R", "val_entropy"])

    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root   = "data"
    env_folders = sorted([d for d in os.listdir(data_root) if d.isdigit()], key=int)

    # -----------------  loop over environments  ------------------
    for env_name in env_folders:
        env_path = os.path.join(data_root, env_name)
        raw = load_data_for_environment(env_path)
        if not raw:
            print(f"[Skip] No data in {env_path}")
            continue

        dataset = SubgoalDataset(raw, device=device)
        train_size = int(0.8 * len(dataset))
        val_size   = len(dataset) - train_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                                  shuffle=True,  collate_fn=custom_collate_fn)
        val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                                  shuffle=True,  collate_fn=custom_collate_fn)

        # --------------- “epochs” just repeat rule evaluation ----
        for epoch in range(1, NUM_EPOCHS_PER_ENV + 1):

            # ─── TRAIN-phase stats (rule is deterministic, but we log) ──
            train_R, train_H = [], []
            for batch in train_loader:
                tau      = batch['Ot_to_goal'].to(device).squeeze(-1)     # (B,)
                probs    = rule_probs_temp(tau)                    # (B,10)
                entropy  = -(probs * probs.log()).sum(dim=-1)            # (B,)

                R_scalar = gini_reward(probs, tau, ALPHA_R)              # (B,)
                train_R.append(R_scalar.mean().item())
                train_H.append(entropy.mean().item())

            # ─── VALIDATION every 10 “epochs”  ────────────────────────
            if epoch % 1 == 0:
                val_R, val_H = [], []
                with torch.no_grad():
                    for b_idx, batch in enumerate(val_loader):
                        B = len(batch['Ot_to_goal'])
                        tau_v   = batch['Ot_to_goal'].to(device).squeeze(-1)
                        probs_v = rule_probs_temp(tau_v)
                        entropy_v = -(probs_v * probs_v.log()).sum(dim=-1)
                        R_val = gini_reward(probs_v, tau_v, ALPHA_R)

                        val_R.append(R_val.mean().item())
                        val_H.append(entropy_v.mean().item())

                        # one nice heat-map
                        if b_idx == 0:
                            sample_map = batch['band_idx_map'].view(B, 100, 100)[0]
                            paths_pos   = batch['paths_positions'][0]
                            sp          = batch['start_point'][0]
                            ot          = batch['Ot_to_goal'][0].item()
                            # print(f"τ range in batch → min {tau.min():.2f}  max {tau.max():.2f}")
                            visualize_band_probs(
                                sample_map.cpu(), probs_v[0].cpu().numpy(),
                                paths_pos, sp,
                                f"[RULE] Env {env_name} Ep {epoch}  τ={ot:.2f}",
                                os.path.join(run_dir, f"rule_env{env_name}_ep{epoch}.png"))

                # CSV log
                with open(csv_path, "a", newline="") as f:
                    csv.writer(f).writerow([env_name, epoch,
                                            sum(val_R)/len(val_R),
                                            sum(val_H)/len(val_H)])

    print("\nFinished rule-based run ✨")

# ------------------------------------------------------------------
if __name__ == "__main__":
    main()
