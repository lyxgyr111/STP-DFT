import re
import matplotlib.pyplot as plt
log_file_path = 'full_dft_training_log.txt'
output_image_file = 'direction2_loss_curve.png'
curve_label = 'Full DFT Model (Direction 2)'
plt.figure(figsize=(12, 8))
print(f"Processing log file: {log_file_path}")
steps = []
val_losses = []
regex = re.compile(r"step (\d+):.*?val loss ([\d.]+)")
try:
    with open(log_file_path, 'r', encoding='utf-16-le') as f:
        for line in f:
            match = regex.search(line)
            if match:
                step = int(match.group(1))
                val_loss = float(match.group(2))
                if step > 0:
                    steps.append(step)
                    val_losses.append(val_loss)
    if steps:
        plt.plot(steps, val_losses, marker='o', linestyle='-', label=curve_label)
        print(f"  -> Found {len(steps)} data points. Plotted successfully.")
    else:
        print(f"  -> No data points found in {log_file_path}.")
except FileNotFoundError:
    print(
        f"  -> ERROR: Log file '{log_file_path}' not found. Please make sure the training has run and the log file exists.")
plt.title('Validation Loss for Full DFT Model', fontsize=16)
plt.xlabel('Training Steps', fontsize=12)
plt.ylabel('Validation Loss', fontsize=12)
plt.grid(True)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(output_image_file)
print(f"\nLoss curve has been saved to '{output_image_file}'")