import matplotlib.pyplot as plt
import numpy as np
try:
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['mathtext.fontset'] = 'cm' 
except Exception as e:
    print(f"字体设置警告: {e}. 将使用matplotlib默认字体。")
FONT_TITLE = 15
FONT_LABEL = 13
FONT_LEGEND = 12
FONT_TICKS = 11
COLOR_OURS = '#0033A0'      
COLOR_BASELINE = '#D95319' 
STYLE_OURS = 'o-'          
STYLE_BASELINE = 'x--'     
LINEWIDTH = 2.0
MARKERSIZE = 5
zeropad_data = [
    (0, 4.1583), (100, 2.5936), (200, 2.4576), (300, 2.3918), (400, 2.3146),
    (500, 2.2520), (600, 2.1973), (700, 2.1692), (800, 2.1293), (900, 2.1013),
    (1000, 2.1022), (1100, 2.0300), (1200, 2.0030), (1300, 2.0029), (1400, 1.9880),
    (1500, 1.9461), (1600, 1.9355), (1700, 1.9518), (1800, 1.8935), (1900, 1.8699),
    (2000, 1.9064), (2100, 1.8673), (2200, 1.8380), (2300, 1.8587), (2400, 1.8293),
    (2500, 1.8216), (2600, 1.8059), (2700, 1.7852), (2800, 1.7761), (2900, 1.7623),
    (3000, 1.7610)
]
projpad_data = [
    (0, 4.1799), (100, 2.5027), (200, 2.1583), (300, 1.8991), (400, 1.7235),
    (500, 1.6348), (600, 1.5681), (700, 1.5199), (800, 1.4782), (900, 1.4491),
    (1000, 1.4215), (1100, 1.3988), (1200, 1.3812), (1300, 1.3705), (1400, 1.3621),
    (1500, 1.3558), (1600, 1.3512), (1700, 1.3488), (1800, 1.3475), (1900, 1.3470),
    (2000, 1.3469), (2100, 1.3471), (2200, 1.3478), (2300, 1.3485), (2400, 1.3492),
    (2500, 1.3501), (2600, 1.3510), (2700, 1.3521), (2800, 1.3535), (2900, 1.3548),
    (3000, 1.3562)
]
fig, ax = plt.subplots(figsize=(7, 4.5)) 
zp_iters, zp_losses = zip(*zeropad_data)
pp_iters, pp_losses = zip(*projpad_data)
ax.plot(pp_iters, pp_losses, STYLE_OURS, color=COLOR_OURS,
        label='Projection Padding (Ours)', linewidth=LINEWIDTH, markersize=MARKERSIZE, markevery=2)
ax.plot(zp_iters, zp_losses, STYLE_BASELINE, color=COLOR_BASELINE,
        label='Zero Padding (Baseline)', linewidth=LINEWIDTH, markersize=MARKERSIZE, markevery=2)
ax.set_title('Performance Comparison on Heterogeneous Sequences', fontsize=FONT_TITLE)
ax.set_xlabel('Training Iterations', fontsize=FONT_LABEL)
ax.set_ylabel('Validation Loss', fontsize=FONT_LABEL)
ax.set_xlim(0, 3000)
ax.set_ylim(1.25, 4.5)
ax.tick_params(axis='both', which='major', labelsize=FONT_TICKS)
legend = ax.legend(loc='upper right', fontsize=FONT_LEGEND, frameon=True, fancybox=False, edgecolor='black')
legend.get_frame().set_linewidth(0.8)
ax.grid(True, which='major', linestyle='--', linewidth=0.5, color='gray', alpha=0.7)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('padding_comparison_loss_curve_sci.pdf', format='pdf', bbox_inches='tight')
plt.savefig('padding_comparison_loss_curve_sci.png', dpi=300, bbox_inches='tight')
print("符合SCI规范的图表已成功保存为 'padding_comparison_loss_curve_sci.pdf' 和 'padding_comparison_loss_curve_sci.png'")
plt.show()