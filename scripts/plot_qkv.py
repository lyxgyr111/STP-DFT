import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Palatino', 'Georgia'],
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.dpi': 300,
})
model_configs = [
    'Baseline\n(128-128-128)',
    'Shrink-K\n(128-64-128)',
    'Expand-V\n(128-64-192)'
]
params_m = np.array([0.829, 0.771, 0.837])
val_losses = np.array([0.5797, 0.5164, 0.4436])
x = np.arange(len(model_configs))
width = 0.4
fig, ax1 = plt.subplots(figsize=(8, 5))
color_params = 'cornflowerblue'
bars = ax1.bar(x, params_m, width, label='Total Parameters (Millions)', color=color_params,
               edgecolor='black', linewidth=1.0)
ax1.set_ylabel('Total Parameters (Millions)', fontweight='bold')
ax1.tick_params(axis='y', labelcolor='black')
ax1.set_ylim(0, max(params_m) * 1.35)
ax1.set_xlabel('Model Configuration', fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(model_configs)
ax1.tick_params(axis='x', length=0)
ax2 = ax1.twinx()
color_loss = 'crimson'
ax2.plot(x, val_losses, color=color_loss, marker='o', linestyle='--',
         linewidth=2, markersize=7, label='Best Validation Loss')
ax2.set_ylabel('Best Validation Loss', fontweight='bold')
ax2.tick_params(axis='y', labelcolor='black')
ax2.set_ylim(min(val_losses) * 0.95, max(val_losses) * 1.05)
ax1.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax1.set_axisbelow(True)
ax1.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.5)
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper center', frameon=True, fancybox=True, shadow=False, framealpha=0.9, ncol=2)
fig.tight_layout()
plt.savefig('dft_parameter_efficiency_publishable_v1.pdf', bbox_inches='tight')
plt.savefig('dft_parameter_efficiency_publishable_v1.png', dpi=300, bbox_inches='tight')
print("Publication-ready chart (V1) saved as 'dft_parameter_efficiency_publishable_v1.pdf' and '.png'")
plt.show()