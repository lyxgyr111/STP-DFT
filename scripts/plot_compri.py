import re
import matplotlib.pyplot as plt
import argparse
import numpy as np
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
def parse_log_file(file_path):
    """解析单个日志文件，返回验证损失字典。"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            log_data = f.read()
    except FileNotFoundError:
        print(f"错误：找不到文件 {file_path}")
        return None
    val_loss_pattern = re.compile(r"step (\d+):.*?val loss ([\d.]+)")
    val_losses = {}
    for line in log_data.strip().split('\n'):
        match = val_loss_pattern.search(line)
        if match:
            step = int(match.group(1))
            loss = float(match.group(2))
            val_losses[step] = loss
    return val_losses
def plot_comparison(log1_path, label1, log2_path, label2):
    """
    绘制两个日志文件的验证损失曲线对比图，并增加一个差值图。
    """
    val1 = parse_log_file(log1_path)  
    val2 = parse_log_file(log2_path)  
    if val1 is None or val2 is None:
        print("因文件读取失败，无法生成图表。")
        return
    if not val1 or not val2:
        print("错误：一个或两个日志文件中没有找到任何'val loss'数据。请检查日志内容。")
        return
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True,
                             gridspec_kw={'height_ratios': [3, 1]})
    ax_main = axes[0]
    ax_diff = axes[1]
    sorted_iters1 = sorted(val1.keys())
    sorted_losses1 = [val1[i] for i in sorted_iters1]
    ax_main.plot(sorted_iters1, sorted_losses1, 'o-', label=f'{label1} (验证损失)', color='crimson', markersize=7,
                 linewidth=2)
    sorted_iters2 = sorted(val2.keys())
    sorted_losses2 = [val2[i] for i in sorted_iters2]
    ax_main.plot(sorted_iters2, sorted_losses2, 's--', label=f'{label2} (验证损失)', color='royalblue', markersize=6,
                 linewidth=2)
    ax_main.set_title('消融实验：Project 投影 vs. Zero_Pad 零填充', fontsize=20, fontweight='bold')
    ax_main.set_ylabel('验证损失 (Validation Loss)', fontsize=14)
    ax_main.legend(fontsize=12)
    ax_main.grid(True, which='major', linestyle='--', alpha=0.7)
    ax_main.minorticks_on()
    ax_main.grid(True, which='minor', linestyle=':', alpha=0.4)
    all_losses = sorted_losses1 + sorted_losses2
    if all_losses:
        plot_min = min(all_losses)
        plot_max = max(all_losses)
        padding = (plot_max - plot_min) * 0.1  
        ax_main.set_ylim(plot_min - padding, plot_max + padding)
    common_steps = sorted(list(set(val1.keys()) & set(val2.keys())))
    differences = [val2[step] - val1[step] for step in common_steps]
    ax_diff.plot(common_steps, differences, 'o-', color='green', label='差值 (标准 - DFT)')
    ax_diff.axhline(0, color='black', linestyle='--', linewidth=1)
    ax_diff.set_xlabel('训练迭代次数 (Iterations)', fontsize=14)
    ax_diff.set_ylabel('损失差值', fontsize=14)
    ax_diff.grid(True, linestyle='--', alpha=0.7)
    ax_diff.text(0.5, 0.9, '差值 > 0: 投影处理序列更优', transform=ax_diff.transAxes,
                 ha='center', va='top', fontsize=10, color='blue')
    ax_diff.text(0.5, 0.1, '差值 < 0: 零填充更优', transform=ax_diff.transAxes,
                 ha='center', va='bottom', fontsize=10, color='red')
    fig.tight_layout(pad=2.0)  
    print("\n正在生成优化后的对比图表...")
    plt.show()
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='比较两个模型训练日志的损失曲线，并显示其差值。')
    parser.add_argument('project_log', help='dft.txt')
    parser.add_argument('zeroed_log', help='lingtianchong.txt')
    args = parser.parse_args()
    plot_comparison(args.project_log, '投影', args.zeroed_log, '零填充')