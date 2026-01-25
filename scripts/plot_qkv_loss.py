import re
import matplotlib.pyplot as plt
import numpy as np
def parse_log_file(file_path):
    """
    从指定的日志文件中解析出 step 和 val_loss 数据。
    """
    steps = []
    val_losses = []
    pattern = re.compile(r"step (\d+):.*?val loss ([\d\.]+)")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    steps.append(int(match.group(1)))
                    val_losses.append(float(match.group(2)))
    except FileNotFoundError:
        print(f"错误: 文件 '{file_path}' 未找到。请确保文件名正确且文件在同一目录下。")
        return None, None
    return steps, val_losses
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "axes.linewidth": 1.5
})
baseline_steps, baseline_losses = parse_log_file('baseline_log.txt')
shrink_k_steps, shrink_k_losses = parse_log_file('shrink_k_log.txt')
expand_v_steps, expand_v_losses = parse_log_file('expand_v_log.txt')
if not all([baseline_steps, shrink_k_steps, expand_v_steps]):
    print("一个或多个日志文件解析失败，请检查错误信息后重试。")
else:
    fig, ax = plt.subplots(figsize=(10, 7))
    styles = {
        'Baseline': {'color': 'black', 'marker': 'o', 'linestyle': '-'},
        'Shrink-K': {'color': 'dimgray', 'marker': 's', 'linestyle': '--'},
        'Expand-V': {'color': 'firebrick', 'marker': '^', 'linestyle': '-'}
    }
    ax.plot(baseline_steps, baseline_losses,
            label='Baseline (128-128-128)',
            linewidth=2, markersize=6, markevery=5, **styles['Baseline'])
    ax.plot(shrink_k_steps, shrink_k_losses,
            label='Shrink-K (128-64-128)',
            linewidth=2, markersize=6, markevery=5, **styles['Shrink-K'])
    ax.plot(expand_v_steps, expand_v_losses,
            label='Expand-V (128-64-192)',
            linewidth=2.5, markersize=7, markevery=5, **styles['Expand-V'])
    ax.set_title('Validation Loss Convergence Comparison', fontweight='bold')
    ax.set_xlabel('Training Iterations (Steps)', fontweight='bold')
    ax.set_ylabel('Validation Loss (Lower is Better)', fontweight='bold')
    ax.legend(loc='upper right', frameon=True, edgecolor='black')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(0, max(baseline_steps) * 1.05)
    min_loss = min(min(baseline_losses), min(shrink_k_losses), min(expand_v_losses))
    max_loss = max(baseline_losses[0], shrink_k_losses[0], expand_v_losses[0])
    ax.set_ylim(min_loss * 0.95, max_loss * 1.05)
    best_loss_val = min(expand_v_losses)
    best_loss_idx = np.argmin(expand_v_losses)
    best_loss_step = expand_v_steps[best_loss_idx]
    ax.plot(best_loss_step, best_loss_val, marker='^', markersize=12,
            color='crimson', markeredgecolor='black', zorder=10)
    ax.annotate(f'Best Performance\nLoss: {best_loss_val:.4f}',
                xy=(best_loss_step, best_loss_val),  
                xytext=(best_loss_step - 600, best_loss_val + 0.4),
                arrowprops=dict(arrowstyle="-", color='black', connectionstyle="arc3,rad=0.3"),  
                fontsize=12,
                ha='center'  
                )
    plt.tight_layout()
    plt.savefig('dft_loss_curves_comparison_academic_final.pdf', bbox_inches='tight')
    plt.savefig('dft_loss_curves_comparison_academic_final.png', dpi=300, bbox_inches='tight')
    print("Final academic-style chart saved as 'dft_loss_curves_comparison_academic_final.pdf' and '.png'")
    plt.show()