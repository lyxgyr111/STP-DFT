import re
import matplotlib.pyplot as plt
import argparse
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
def parse_log_file(file_path):
    """解析单个日志文件，返回训练和验证损失字典。"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            log_data = f.read()
    except FileNotFoundError:
        print(f"错误：找不到文件 {file_path}")
        return None, None
    train_loss_pattern = re.compile(r"step (\d+): train loss ([\d.]+)")
    val_loss_pattern = re.compile(r"step (\d+):.*?val loss ([\d.]+)")
    train_losses = {}
    val_losses = {}
    for line in log_data.strip().split('\n'):
        train_match = train_loss_pattern.search(line)
        val_match = val_loss_pattern.search(line)
        if train_match:
            step = int(train_match.group(1))
            loss = float(train_match.group(2))
            train_losses[step] = loss
        if val_match:
            step = int(val_match.group(1))
            loss = float(val_match.group(2))
            val_losses[step] = loss
    return train_losses, val_losses
def plot_single_log(log_path, model_name):
    """绘制单个日志文件的损失曲线图。"""
    train_losses, val_losses = parse_log_file(log_path)
    if not train_losses and not val_losses:
        print(f"错误：在 {log_path} 中未找到任何损失数据。")
        return
    fig, ax = plt.subplots(figsize=(14, 8))
    if train_losses:
        sorted_iters = sorted(train_losses.keys())
        sorted_losses = [train_losses[i] for i in sorted_iters]
        ax.plot(sorted_iters, sorted_losses, 'o-', label=f'{model_name} (训练损失)', color='royalblue', alpha=0.8)
    if val_losses:
        sorted_iters = sorted(val_losses.keys())
        sorted_losses = [val_losses[i] for i in sorted_iters]
        ax.plot(sorted_iters, sorted_losses, 's--', label=f'{model_name} (验证损失)', color='crimson', markersize=8)
    ax.set_title(f'{model_name} 性能曲线 (固定维度任务)', fontsize=18, fontweight='bold')
    ax.set_xlabel('训练迭代次数 (Iterations)', fontsize=14)
    ax.set_ylabel('损失 (Loss)', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    print(f"\n正在为 {model_name} 生成图表...")
    plt.show()
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='绘制单个模型训练日志的损失曲线。')
    parser.add_argument('log_file', help='baseline.txt)')
    parser.add_argument('--name', default='nanoGPT 模型', help='图表中显示的模型名称')
    args = parser.parse_args()
    plot_single_log(args.log_file, args.name)