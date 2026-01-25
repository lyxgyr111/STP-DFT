import matplotlib.pyplot as plt
import numpy as np
labels = ['Standard Head\n[32, 32, 32, 32]', 'Mixed Head\n[64, 32, 16, 16]']
parameters = [0.829, 0.829]  
val_loss = [0.5786, 0.5586]
x = np.arange(len(labels))  
width = 0.45  
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.figsize": (7, 5), 
})
fig, ax1 = plt.subplots()
color_params = '#4682B4' 
bars = ax1.bar(x, parameters, width, label='Total Parameters (M)', color=color_params, alpha=0.8, edgecolor='black')
ax2 = ax1.twinx()
color_loss = '#D55E00' 
line = ax2.plot(x, val_loss, color=color_loss, marker='D', linestyle='--', linewidth=2.5, markersize=8, label='Best Validation Loss')
ax1.set_title('Performance vs. Parameters', fontweight='bold')
ax1.set_xlabel('Model Configuration', fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.set_ylabel('Total Parameters (M)', fontweight='bold', color=color_params)
ax1.tick_params(axis='y', labelcolor=color_params)
ax1.set_ylim(0, 1.0) 
for bar in bars:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.3f}M',
             ha='center', va='bottom', fontsize=11, fontweight='bold',
             bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.2'))
ax2.set_ylabel('Best Validation Loss', fontweight='bold', color=color_loss)
ax2.tick_params(axis='y', labelcolor=color_loss)
ax2.set_ylim(0.55, 0.59)
for i, txt in enumerate(val_loss):
    ax2.annotate(f'{txt:.4f}', (x[i], val_loss[i]), textcoords="offset points",
                 xytext=(0, 12), ha='center', fontsize=11, fontweight='bold', color=color_loss)
lines, labels_line = ax1.get_legend_handles_labels()
lines2, labels_line2 = ax2.get_legend_handles_labels()
legend = ax2.legend(lines + lines2, labels_line + labels_line2, loc='upper center',
                    bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=False, ncol=2)
legend.get_frame().set_edgecolor('black')
ax1.grid(False) 
ax2.grid(False)
ax1.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)
fig.tight_layout(rect=[0, 0.1, 1, 1])
output_filename = 'performance_comparison.pdf'
plt.savefig(output_filename, format='pdf', bbox_inches='tight')
print(f"SCI-quality plot saved as '{output_filename}'")
plt.show()