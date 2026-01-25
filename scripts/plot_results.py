import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.family'] = 'Times New Roman' 
plt.rcParams['font.size'] = 14                  
plt.rcParams['axes.labelsize'] = 16             
plt.rcParams['xtick.labelsize'] = 12            
plt.rcParams['ytick.labelsize'] = 12            
plt.rcParams['legend.fontsize'] = 14            
sns.set_theme(style="whitegrid", font="Times New Roman")
steps = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000, 3100, 3200, 3300, 3400, 3500, 3600, 3700, 3800, 3900, 4000, 4100, 4200, 4300, 4400, 4500, 4600, 4700, 4800, 4900, 5000]
train_loss_standard_gpt = [4.1915, 3.4541, 3.0243, 2.7283, 2.5663, 2.4664, 2.3637, 2.2824, 2.2009, 2.0922, 2.0043, 1.9277, 1.8499, 1.7821, 1.7176, 1.6523, 1.6030, 1.5549, 1.5185, 1.4684, 1.4345, 1.4023, 1.3697, 1.3479, 1.3205, 1.3060, 1.2856, 1.2707, 1.2600, 1.2468, 1.2283, 1.2145, 1.2042, 1.1965, 1.1862, 1.1775, 1.1671, 1.1603, 1.1539, 1.1453, 1.1394, 1.1331, 1.1275, 1.1222, 1.1165, 1.1130, 1.1100, 1.1054, 1.1036, 1.1017, 1.1013]
val_loss_standard_gpt = [4.1902, 3.4656, 3.0402, 2.7424, 2.5794, 2.4771, 2.3791, 2.3076, 2.2314, 2.1378, 2.0719, 2.0060, 1.9581, 1.9160, 1.8663, 1.8173, 1.7706, 1.7321, 1.7040, 1.6595, 1.6351, 1.6052, 1.5965, 1.5785, 1.5590, 1.5524, 1.5446, 1.5379, 1.5322, 1.5390, 1.5315, 1.5276, 1.5259, 1.5312, 1.5296, 1.5366, 1.5408, 1.5386, 1.5399, 1.5365, 1.5454, 1.5476, 1.5513, 1.5572, 1.5504, 1.5572, 1.5625, 1.5590, 1.5716, 1.5672, 1.5695]
train_loss_dft_model = [4.1741, 2.5645, 2.4618, 2.3867, 2.3104, 2.2439, 2.1461, 2.0389, 1.9529, 1.8629, 1.8076, 1.7441, 1.6954, 1.6415, 1.6226, 1.5904, 1.5580, 1.5445, 1.5212, 1.5098, 1.4828, 1.4721, 1.4566, 1.4398, 1.4267, 1.4261, 1.4083, 1.4090, 1.3953, 1.3821, 1.3720, 1.3627, 1.3588, 1.3525, 1.3433, 1.3374, 1.3238, 1.3144, 1.3134, 1.3103, 1.2994, 1.3006, 1.2969, 1.2964, 1.2875, 1.2902, 1.2786, 1.2750, 1.2705, 1.2787, 1.2729]
val_loss_dft_model = [4.1689, 2.5621, 2.4713, 2.4028, 2.3267, 2.2796, 2.2011, 2.1175, 2.0471, 1.9960, 1.9498, 1.9019, 1.8635, 1.8089, 1.7827, 1.7725, 1.7439, 1.7116, 1.7120, 1.6978, 1.6833, 1.6757, 1.6582, 1.6436, 1.6297, 1.6279, 1.6149, 1.6180, 1.6023, 1.6047, 1.5865, 1.5740, 1.5878, 1.5778, 1.5777, 1.5689, 1.5628, 1.5462, 1.5573, 1.5613, 1.5493, 1.5418, 1.5457, 1.5344, 1.5360, 1.5262, 1.5430, 1.5289, 1.5308, 1.5326, 1.5358]
plt.figure(figsize=(10, 7)) 
plt.plot(steps, train_loss_standard_gpt, color='royalblue', linestyle=':', alpha=0.9, label='Standard GPT - Train Loss')
plt.plot(steps, val_loss_standard_gpt, color='darkorange', linestyle='-', linewidth=2.5, label='Standard GPT - Validation Loss')
plt.plot(steps, train_loss_dft_model, color='seagreen', linestyle=':', alpha=0.9, label='DFT Model - Train Loss')
plt.plot(steps, val_loss_dft_model, color='crimson', linestyle='-', linewidth=2.5, label='DFT Model - Validation Loss')
plt.title('Loss Curves Comparison on the Shakespeare Dataset', fontsize=20, fontweight='bold')
plt.xlabel('Training Iterations (Steps)')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.ylim(1.0, 4.5)
plt.xlim(0, 5000)
plt.tight_layout()
output_filename_pdf = 'Standard_vs_DFT_Loss_Comparison.pdf'
plt.savefig(output_filename_pdf, format='pdf', bbox_inches='tight')
print(f"矢量图 (首选) 已保存为: {output_filename_pdf}")
output_filename_png = 'Standard_vs_DFT_Loss_Comparison.png'
plt.savefig(output_filename_png, dpi=300, bbox_inches='tight')
print(f"高分辨率位图 (备用) 已保存为: {output_filename_png}")
plt.show()