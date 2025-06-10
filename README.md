# 🎐 Fixed-Entropy Band Sampler

This repo hosts an *ultra-light* **replacement** for my neural-network **BandNet**.  
Instead of training, it **maps the normalised distance‐to-goal (τ)** directly to a softmax temperature and returns a probability over 10 radial bands:

τ → T(τ) → softmax( −band_idx / T )

* Near goal (τ → 0) ➜ tiny T ➜ sharp focus on bands near the global path.  
* Far away (τ → 1) ➜ large T ➜ broad exploration across the corridor.

## 🌟 Why I chose this method

* **Zero training time** – perfect baseline or quick deployment.
* **Guaranteed entropy control** – no more over- or under-exploration.
* **One parameter** (`T_min`, `T_max`) to tune the spice level.

## 📂 Project layout

```bash
.
├─ run.py # main entry – loops through all environments
├─ utils/
│ ├─ data_loader.py # unchanged fancy dataloader
│ ├─ environment_functions.py
│ └─ New_Reward_Function.py
└─ results/
└─ data/ # I didn't upload it due to its size
```

## 🚀 Quick start

```bash
# clone & enter
git clone git@github.com:ZhangJingru-Ruby/FixedEntropyBandRule.git
cd FixedEntropyBandRule

# (optionally) activate your conda env
conda activate PY38-ZJR-subnn

# run!
python run.py
```

Outputs:
CSV log of reward + entropy per epoch.

Heat-map PNGs every 10th epoch (results/rule_*/rule_envX_epY.png)

## ⚙️ Key hyper-params

| Variable  | Default | Meaning                               |
| --------- | ------- | ------------------------------------- |
| `T_min`   | `0.05`  | Focused spread at τ=0                 |
| `T_max`   | `6.0`   | Exploratory spread at τ=1             |
| `ALPHA_R` | `5.0`   | Reward scale (same as BandNet papers) |



