import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
loss_A_raw = {0:4.1799, 100:3.7756, 200:3.3713, 300:2.9670, 400:2.5627, 500:2.1584,
              1000:0.9512, 1500:0.7954, 2000:0.7002, 2500:0.6595, 3000:0.5786}
steps_A_keys = list(loss_A_raw.keys())
loss_A_values = list(loss_A_raw.values())
cs = CubicSpline(steps_A_keys, loss_A_values)
steps_A_smooth = np.linspace(0, 3000, 500) 
loss_A_smooth = cs(steps_A_smooth)
steps_B = np.arange(0, 3001, 100)
loss_B = [
    4.1799, 2.5027, 2.0559, 1.8330, 1.6464, 1.4638, 1.3440, 1.2303, 1.1239, 1.0989,
    1.0665, 0.9705, 0.8827, 0.8820, 0.7846, 0.8231, 0.7934, 0.7751, 0.7110, 0.7416,
    0.7508, 0.7225, 0.6830, 0.6406, 0.6419, 0.5937, 0.6462, 0.5882, 0.6158, 0.6210,
    0.5586
]
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.figsize": (8, 5), 
    "lines.linewidth": 2,
    "lines.markersize": 6,
})
fig, ax = plt.subplots()
ax.plot(steps_A_smooth, loss_A_smooth, linestyle='--', color='#0072B2',
        label='Standard Head [32, 32, 32, 32]')
ax.plot(steps_A_keys, loss_A_values, marker='s', linestyle='none', color='#0072B2')
ax.plot(steps_B, loss_B, marker='o', linestyle='-', color='#D55E00',
        label='Mixed Head [64, 32, 16, 16]', markevery=3)
ax.set_title('Comparison of Learning Dynamics', fontweight='bold')
ax.set_xlabel('Training Steps', fontweight='bold')
ax.set_ylabel('Validation Loss', fontweight='bold')
legend = ax.legend(loc='upper right', frameon=True, framealpha=0.9)
legend.get_frame().set_edgecolor('black')
ax.set_xlim(0, 3000)
ax.set_ylim(0.5, 4.5) 
ax.grid(True, which='major', linestyle='--', linewidth=0.5, color='gray')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
output_filename = 'loss_curve_comparison.pdf'
plt.savefig(output_filename, format='pdf', bbox_inches='tight')
print(f"SCI-quality plot saved as '{output_filename}'")
plt.show()