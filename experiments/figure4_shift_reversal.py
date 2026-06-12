"""
Figure 4: NLL trajectories under increasing Gaussian covariate shift.
Four panels show one representative model each (one GBDT, one alternative GBDT,
one single-network neural model, one deep ensemble). Each panel plots median
NLL across the 36 datasets versus shift intensity sigma on x-axis. Five lines
per panel = five calibrators. The in-distribution value (sigma=0) is computed
from results_raw.csv; sigma>0 values from ood_results-2.csv.

Inputs:  results_raw.csv, ood_results-2.csv
Outputs: figure4_shift_reversal.pdf, figure4_shift_reversal.png
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------- data ----------------------------------------------------------------
df_id  = pd.read_csv("results_raw.csv")
df_ood = pd.read_csv("ood_results-2.csv")

# In-distribution (sigma=0): per-dataset median across 5 seeds, then median across datasets
id_med = (df_id.groupby(['task_id','model','calibrator'])
                ['cal_nll'].median().reset_index()
                .groupby(['model','calibrator'])['cal_nll'].median())

# OOD (sigma in {0.5, 1.0, 2.0}): single seed, median across datasets
ood_med = (df_ood.groupby(['model','calibrator','ood_sigma'])
                  ['cal_nll'].median().reset_index())

models_to_show = [
    ('lgbm',          'LightGBM'),
    ('catboost',      'CatBoost'),
    ('single_mlp',    'SingleMLP'),
    ('deep_ensemble', 'Deep Ensemble ($M{=}5$)'),
]

cal_order   = ['none', 'temp', 'logistic', 'isotonic', 'dirichlet']
cal_labels  = ['Raw', 'Temp. Scaling', 'MLR', 'Isotonic Reg.', 'Dirichlet Cal.']
cal_colors  = {
    'none':      '#7F8C8D',  # gray
    'temp':      '#27AE60',  # green (good in-dist)
    'logistic':  '#C0392B',  # red (bad in-dist → good OOD)
    'isotonic':  '#8E44AD',  # purple
    'dirichlet': '#2980B9',  # blue
}
cal_markers = {'none':'o', 'temp':'s', 'logistic':'D', 'isotonic':'^', 'dirichlet':'v'}

# Sigma values to plot, with sigma=0 being in-distribution
sigmas = [0.0, 0.5, 1.0, 2.0]

# ------- styling -------------------------------------------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 13,
})
fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))

for ax, (model_key, model_label) in zip(axes.flatten(), models_to_show):
    for cal in cal_order:
        ys = []
        for s in sigmas:
            if s == 0.0:
                try:
                    ys.append(id_med.loc[(model_key, cal)])
                except KeyError:
                    ys.append(np.nan)
            else:
                row = ood_med[(ood_med.model == model_key)
                              & (ood_med.calibrator == cal)
                              & (ood_med.ood_sigma == s)]
                ys.append(row['cal_nll'].values[0] if len(row) else np.nan)
        ax.plot(sigmas, ys, marker=cal_markers[cal], markersize=9,
                linewidth=2.2, color=cal_colors[cal],
                label=cal_labels[cal_order.index(cal)])

    ax.set_xticks(sigmas)
    ax.set_xticklabels([f"$\\sigma{{=}}{s}$\n" + ("in-dist." if s == 0 else "") for s in sigmas])
    ax.set_xlabel("Gaussian feature-shift intensity", fontsize=12.5)
    ax.set_ylabel("Median NLL across 36 datasets", fontsize=12.5)
    ax.set_title(model_label, fontsize=14, weight='bold', loc='left')
    ax.tick_params(labelsize=11.5)
    ax.grid(True, alpha=0.25, linestyle=':')
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)

# Shared legend at the top of the figure
handles, labels = axes[0,0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=5, fontsize=12.5,
           frameon=True, bbox_to_anchor=(0.5, 1.005), borderpad=0.6,
           columnspacing=1.6, handletextpad=0.6)

fig.suptitle("Calibrator ordering reverses under covariate shift\n"
             "(median NLL vs.\\ Gaussian feature-shift intensity; $n = 36$ datasets per point)",
             fontsize=14.5, y=1.07, weight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("figure4_shift_reversal.pdf", bbox_inches="tight", pad_inches=0.05)
plt.savefig("figure4_shift_reversal.png", bbox_inches="tight", pad_inches=0.05, dpi=200)
print("saved figure4_shift_reversal.{pdf,png}")
