import re
import numpy as np
def analyze_log_time(log_file_path):
    """
    解析训练日志文件，自动检测文件编码，提取每次迭代的时间，
    并计算平均值、标准差等统计数据。
    Args:
        log_file_path (str): 日志文件的路径。
    """
    iteration_times = []
    time_pattern = re.compile(r"time ([\d.]+)ms")
    encodings_to_try = ['utf-8', 'utf-8-sig', 'utf-16', 'latin-1']
    file_content = None
    for encoding in encodings_to_try:
        try:
            with open(log_file_path, 'r', encoding=encoding) as f:
                file_content = f.readlines()
            print(f"成功使用 '{encoding}' 编码读取文件。")
            break  
        except UnicodeDecodeError:
            continue
        except FileNotFoundError:
            print(f"错误: 文件 '{log_file_path}' 未找到。")
            return
    if file_content is None:
        print(f"错误: 尝试了 {encodings_to_try} 所有编码后，仍无法解码文件 '{log_file_path}'。")
        return
    for line in file_content:
        match = time_pattern.search(line)
        if match:
            time_ms = float(match.group(1))
            iteration_times.append(time_ms)
    if not iteration_times:
        print(f"警告: 在文件 '{log_file_path}' 中没有找到任何时间数据。")
        return
    times_array = np.array(iteration_times)
    avg_time = np.mean(times_array)
    std_time = np.std(times_array)
    min_time = np.min(times_array)
    max_time = np.max(times_array)
    total_iterations = len(iteration_times)
    print("=" * 40)
    print(f"性能分析报告: {log_file_path}")
    print("=" * 40)
    print(f"总迭代次数: {total_iterations}")
    print(f"平均迭代时间: {avg_time:.2f} ms")
    print(f"时间标准差: {std_time:.2f} ms (衡量稳定性)")
    print(f"最快迭代时间: {min_time:.2f} ms")
    print(f"最慢迭代时间: {max_time:.2f} ms")
    print("=" * 40)
if __name__ == "__main__":
    log_file_to_analyze = 'full_dft_training_log.txt'
    analyze_log_time(log_file_to_analyze)