import json
import os
import itertools
import matplotlib.pyplot as plt
from scipy import stats
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def file_write(llm_command, output, mismatch_summary, json_file_path, txt_file_path):
    # Append to JSON file
    with open(json_file_path, 'r+') as json_file:
        try:
            data = json.load(json_file)
        except json.JSONDecodeError:
            data = []
        data.append({
            "llm_command": llm_command,
            "output": output,
            "mismatch_summary": mismatch_summary
        })
        json_file.seek(0)
        json.dump(data, json_file, indent=4)
    
    # Append to TXT file
    with open(txt_file_path, 'a') as txt_file:
        txt_file.write(f"LLM Command: {llm_command}\n")
        txt_file.write(f"Output: {output}\n")
        txt_file.write(f"Mismatch Summary: {mismatch_summary}\n")
        txt_file.write("\n")

def summary_tests(folder_path):
    basic_errors = ["remove_ingress", "add_ingress", "change_port", "change_protocol", "add_egress"]
    error_combinations = list(itertools.combinations(basic_errors, 2))
    all_errors = basic_errors + ["+".join(comb) for comb in error_combinations]
    
    success_counts = {error: 0 for error in all_errors}
    total_counts = {error: 0 for error in all_errors}
    iteration_counts = {error: 0 for error in all_errors}
    safety_counts = {error: 0 for error in all_errors}
    
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".json"):
            for error in all_errors:
                if file_name.startswith(error + "_result"):
                    file_path = os.path.join(folder_path, file_name)
                    with open(file_path, 'r') as file:
                        data = json.load(file)
                        if data and "mismatch_summary" in data[-1]:
                            total_counts[error] += 1
                            iteration_counts[error] += len(data)
                            if "No mismatches found" in data[-1]["mismatch_summary"]:
                                success_counts[error] += 1
                            
                            # Check safety
                            safe = True
                            previous_mismatch_count = float('inf')
                            for entry in data:
                                mismatch_summary = entry.get("mismatch_summary", "")
                                mismatch_count = mismatch_summary.count("Mismatch")
                                if mismatch_count > previous_mismatch_count:
                                    safe = False
                                    break
                                previous_mismatch_count = mismatch_count
                            if safe:
                                safety_counts[error] += 1
                    break
    
    result = {}
    for error in all_errors:
        total = total_counts[error]
        success = success_counts[error]
        iterations = iteration_counts[error]
        safety = safety_counts[error]
        average_iteration = iterations / total if total > 0 else 0
        successful_rate = success / total if total > 0 else 0
        safety_rate = safety / total if total > 0 else 0
        result[error] = {
            "total_counts": total,
            "success_counts": success,
            "successful_rate": successful_rate,
            "average_iteration": average_iteration,
            "safety_counts": safety,
            "safety_rate": safety_rate
        }
    
    result_file_path = os.path.join(folder_path, "test_results_summary.json")
    with open(result_file_path, 'w') as result_file:
        json.dump(result, result_file, indent=4)
    
    print(f"Results saved to {result_file_path}")

def plot_metrics(folder_path):
    result_file_path = os.path.join(folder_path, "test_results_summary.json")
    
    with open(result_file_path, 'r') as result_file:
        data = json.load(result_file)
    
    labels = list(data.keys())
    success_rates = [data[error]["successful_rate"] * 100 for error in labels]
    safety_rates = [data[error]["safety_rate"] * 100 for error in labels]
    average_iterations = [data[error]["average_iteration"] for error in labels]
    
    sample_sizes = []
    success_sem_values = []
    safety_sem_values = []
    
    for error in labels:
        total = data[error]["total_counts"]
        success = data[error]["success_counts"]
        safety = data[error]["safety_counts"]
        sample_sizes.append(total)
        
        # Calculate SEM for success rates
        success_binary_outcomes = [1] * success + [0] * (total - success)
        success_scipy_sem = stats.sem(success_binary_outcomes, ddof=0) * 100
        success_sem_values.append(success_scipy_sem)
        
        # Calculate SEM for safety rates
        safety_binary_outcomes = [1] * safety + [0] * (total - safety)
        safety_scipy_sem = stats.sem(safety_binary_outcomes, ddof=0) * 100
        safety_sem_values.append(safety_scipy_sem)
    
    # Calculate 95% confidence interval (1.96 * SEM)
    success_error_margins = [1.96 * sem for sem in success_sem_values]
    safety_error_margins = [1.96 * sem for sem in safety_sem_values]

    result_dir = folder_path
    
    # Plot success rates
    plt.figure(figsize=(12, 6))
    bars = plt.bar(labels, success_rates, color='skyblue', yerr=success_error_margins, capsize=5)
    plt.xlabel('Error Combinations')
    plt.ylabel('Success Rate (%)')
    plt.title('Success Rate by Error Combinations')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, max(success_rates) * 1.1)  # Adjust y-axis limit
    # Add error values on top of each bar
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + success_error_margins[i],
                f'±{success_error_margins[i]:.2f}%',
                ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'success_rate.png'), dpi=300)
    plt.close()

    # Plot safety rates
    plt.figure(figsize=(12, 6))
    bars = plt.bar(labels, safety_rates, color='green', yerr=safety_error_margins, capsize=5)
    plt.xlabel('Error Combinations')
    plt.ylabel('Safety Rate (%)')
    plt.title('Safety Rate by Error Combinations')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, max(safety_rates) * 1.1)  # Adjust y-axis limit
    # Add error values on top of each bar
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + safety_error_margins[i],
                f'±{safety_error_margins[i]:.2f}%',
                ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'safety_rate.png'), dpi=300)
    plt.close()

    # Plot average iterations
    plt.figure(figsize=(12, 6))
    bars = plt.bar(labels, average_iterations, color='orange')
    plt.xlabel('Error Combinations')
    plt.ylabel('Average Iterations')
    plt.title('Average Iterations by Error Combinations')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, max(average_iterations) * 1.1)  # Adjust y-axis limit
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'average_iterations.png'), dpi=300)
    plt.close()

    # Combine all three plots into one figure
    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # Success rates
    axs[0].bar(labels, success_rates, color='skyblue', yerr=success_error_margins, capsize=5)
    axs[0].set_ylabel('Success Rate (%)')
    axs[0].set_title('Success Rate by Error Combinations')
    axs[0].set_ylim(0, min(max(success_rates) * 1.5, 100))  # Adjust y-axis limit

    # Safety rates
    axs[1].bar(labels, safety_rates, color='green', yerr=safety_error_margins, capsize=5)
    axs[1].set_ylabel('Safety Rate (%)')
    axs[1].set_title('Safety Rate by Error Combinations')
    axs[1].set_ylim(0, min(max(safety_rates) * 1.5, 100))  # Adjust y-axis limit

    # Average iterations
    axs[2].bar(labels, average_iterations, color='orange')
    axs[2].set_xlabel('Error Combinations')
    axs[2].set_ylabel('Average Iterations')
    axs[2].set_title('Average Iterations by Error Combinations')
    axs[2].set_ylim(0, min(max(average_iterations) * 1.5, 15))  # Adjust y-axis limit

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'combined_metrics.png'), dpi=300)
    plt.close()

def plot_correctness(folder_path):
    result_file_path = os.path.join(folder_path, "test_results_summary.json")
    
    with open(result_file_path, 'r') as result_file:
        data = json.load(result_file)
    
    labels = list(data.keys())
    correctness_pass_rates = [data[error]["successful_rate"] * 100 for error in labels]
    sample_sizes = []
    sem_values = []
    
    for error in labels:
        total = data[error]["total_counts"]
        success = data[error]["success_counts"]
        sample_sizes.append(total)
        
        # Calculate SEM using scipy
        binary_outcomes = [1] * success + [0] * (total - success)
        scipy_sem = stats.sem(binary_outcomes, ddof=0) * 100
        sem_values.append(scipy_sem)
    
    # Calculate 95% confidence interval (1.96 * SEM)
    error_margins = [1.96 * sem for sem in sem_values]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(labels, correctness_pass_rates, color='green', yerr=error_margins, capsize=5)
    plt.xlabel('Error Combinations')
    plt.ylabel('Correctness Pass Rate (%)')
    plt.title('Correctness Pass Rate by Error Combinations')
    # Add error values on top of each bar
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + error_margins[i],
                f'±{error_margins[i]:.2f}%',
                ha='center', va='bottom')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, 'correctness_pass_rate.png'), dpi=300)
    plt.close()



def summary_different_agent(directory):
    summary_results = {}

    # Iterate through all folders in the given directory
    for folder in os.listdir(directory):
        folder_path = os.path.join(directory, folder)

        # Ensure it's a directory
        if os.path.isdir(folder_path):
            # Check for subdirectories inside this folder
            subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]

            # If there is exactly one subfolder, assume it contains the JSON file
            if len(subfolders) == 1:
                json_file = os.path.join(folder_path, subfolders[0], "test_results_summary.json")
            else:
                json_file = os.path.join(folder_path, "test_results_summary.json")

            # Check if "test_results_summary.json" exists
            if os.path.exists(json_file):
                with open(json_file, "r") as file:
                    data = json.load(file)

                # Initialize counters
                total_queries = 0
                total_success = 0
                total_safety = 0
                success_rates = []
                safety_rates = []

                # Iterate through all error types in the JSON data
                for key, values in data.items():
                    total_queries += values["total_counts"]
                    total_success += values["success_counts"]
                    total_safety += values["safety_counts"]

                    # Create binary lists for success and safety counts
                    success_binary_outcomes = [1] * values["success_counts"] + [0] * (values["total_counts"] - values["success_counts"])
                    safety_binary_outcomes = [1] * values["safety_counts"] + [0] * (values["total_counts"] - values["safety_counts"])

                    # Calculate standard error of mean (SEM) for success rates
                    if len(success_binary_outcomes) > 1:
                        success_rates.append(stats.sem(success_binary_outcomes, ddof=0) * 100)

                    # Calculate SEM for safety rates
                    if len(safety_binary_outcomes) > 1:
                        safety_rates.append(stats.sem(safety_binary_outcomes, ddof=0) * 100)

                # Compute 95% confidence interval (1.96 * SEM)
                success_margin = 1.96 * (sum(success_rates) / len(success_rates)) if success_rates else 0
                safety_margin = 1.96 * (sum(safety_rates) / len(safety_rates)) if safety_rates else 0

                # Store results for each experiment folder
                summary_results[folder] = {
                    "total_queries": total_queries,
                    "total_success": total_success,
                    "total_safety": total_safety,
                    "success_margin": success_margin,
                    "safety_margin": safety_margin
                }

    # Print and return the summary results
    print(json.dumps(summary_results, indent=4))
    return summary_results

def plot_summary_results(directory_path):
    """
    Reads experiment results from multiple folders, plots success vs. safety, 
    and saves the figure inside the directory.
    
    Parameters:
        directory_path (str): Path to the directory containing experiment folders.
    
    Saves:
        summary_plot.png inside directory_path.
    """
    # Get summary results
    summary_results = summary_different_agent(directory_path)

    # Create figure
    plt.figure(figsize=(10, 7))

    # Iterate through each folder's summary and plot points
    for folder, stats in summary_results.items():
        x = stats["total_success"]  # X-axis: Success
        y = stats["total_safety"]   # Y-axis: Safety
        x_err = stats["success_margin"]  # Error bar for success
        y_err = stats["safety_margin"]   # Error bar for safety
        
        # Plot the point with error bars (cross-like bars)
        plt.errorbar(x, y, xerr=x_err, yerr=y_err, fmt='o', capsize=5, label=folder)

        # Annotate folder names (adjust position for better readability)
        plt.annotate(folder, (x, y), textcoords="offset points", xytext=(5,5), ha='center', fontsize=12, weight='bold')

    # Labels and Title
    plt.xlabel("Total Success", fontsize=14)
    plt.ylabel("Total Safety", fontsize=14)
    plt.title("Success vs. Safety with Confidence Margins", fontsize=16)

    # Grid and Legend
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper left", fontsize=10)

    # Save plot inside the directory
    save_path = os.path.join(directory_path, "summary_plot.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Plot saved at: {save_path}")

import os
import json
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def analyze_and_plot(folder_path, output_image="result_plot.png"):
    # 定义所有错误类别
    all_errors = [
        "remove_ingress", "change_port", "add_egress", 
        "remove_ingress+add_ingress", "remove_ingress+change_port",
        "remove_ingress+change_protocol", "add_ingress+change_port",
        "add_ingress+change_protocol", "change_port+change_protocol",
        "change_port+add_egress", "change_protocol+add_egress"
    ]

    # 初始化数据结构
    stages = [5, 20, 30]
    results = {stage: {'success': [], 'safety': []} for stage in stages}

    # 数据收集阶段
    for error in all_errors:
        # 收集当前类别的文件
        files = []
        pattern = re.compile(rf"^{re.escape(error)}_result_(\d+)\.json$")
        for file_name in os.listdir(folder_path):
            if match := pattern.match(file_name):
                files.append((int(match.group(1)), file_name))
        
        # 按编号排序
        files.sort()
        sorted_files = [f[1] for f in files]

        # 分阶段处理
        for stage in stages:
            if len(sorted_files) < stage:
                continue

            stage_files = sorted_files[:stage]
            stage_success = []
            stage_safety = []

            # 处理每个文件
            for file_name in stage_files:
                file_path = os.path.join(folder_path, file_name)
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        if not data:
                            continue
                        
                        # 记录二元结果
                        last_entry = data[-1]
                        success = 1 if "No mismatches found" in last_entry.get("mismatch_summary", "") else 0
                        
                        # 安全检查
                        safe = 1
                        prev_mismatch = float('inf')
                        for entry in data:
                            mismatch_count = entry.get("mismatch_summary", "").count("Mismatch")
                            if mismatch_count > prev_mismatch:
                                safe = 0
                                break
                            prev_mismatch = mismatch_count
                        
                        stage_success.append(success)
                        stage_safety.append(safe)
                except Exception as e:
                    print(f"Error processing {file_name}: {str(e)}")
                    continue

            # 保存原始二元数据
            results[stage]['success'].extend(stage_success)
            results[stage]['safety'].extend(stage_safety)

    # 统计分析阶段
    plot_data = []
    for stage in stages:
        # 获取二元数据
        success_data = np.array(results[stage]['success'])
        safety_data = np.array(results[stage]['safety'])
        
        # 计算均值和置信区间
        success_mean = np.mean(success_data) * 100  # 转换为百分比
        safety_mean = np.mean(safety_data) * 100
        
        # 使用指定方法计算SEM
        success_sem = stats.sem(success_data, ddof=0) * 100
        safety_sem = stats.sem(safety_data, ddof=0) * 100
        
        success_ci = 1.96 * success_sem
        safety_ci = 1.96 * safety_sem
        
        plot_data.append({
            'stage': stage,
            'x': safety_mean,  # 横轴改为Safety
            'y': success_mean,  # 纵轴改为Success
            'xerr': safety_ci,
            'yerr': success_ci,
            'safety_ci': (safety_mean - safety_ci, safety_mean + safety_ci),
            'success_ci': (success_mean - success_ci, success_mean + success_ci)
        })

    # 可视化阶段
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    plt.rcParams.update({'font.size': 12, 'font.family': 'DejaVu Sans'})
    
    # 设置统一坐标范围
    for ax in axes:
        ax.set_xlim(0, 100)  # Safety百分比范围
        ax.set_ylim(0, 100)  # Success百分比范围
        ax.grid(True, linestyle=':', alpha=0.5)

    for ax, data in zip(axes, plot_data):
        # 绘制误差条（交换坐标轴）
        ax.errorbar(
            x=data['x'], y=data['y'],
            xerr=data['xerr'], yerr=data['yerr'],
            fmt='o', markersize=10, capsize=8,
            color='#2c7bb6', ecolor='#d7191c', 
            markeredgewidth=2, elinewidth=2
        )
        
        # 添加坐标标注
        ax.text(
            data['x'] + 1, data['y'] - 3,
            f"({data['x']:.1f}%, {data['y']:.1f}%)",
            fontsize=10, ha='left', va='top',
            bbox=dict(facecolor='white', alpha=0.8)
        )
        
        # 添加置信区间标注
        ax.text(0.05, 0.15,
               f"Safety CI: [{data['safety_ci'][0]:.1f}%, {data['safety_ci'][1]:.1f}%]\n"
               f"Success CI: [{data['success_ci'][0]:.1f}%, {data['success_ci'][1]:.1f}%]",
               transform=ax.transAxes, fontsize=8,
               bbox=dict(facecolor='white', alpha=0.7))
        
        # 标注样本量和模型
        ax.text(0.05, 0.95, f'N={data["stage"]}', 
                transform=ax.transAxes, va='top', fontweight='bold')
        ax.text(0.95, 0.05, 'gpt-4o-base', 
                transform=ax.transAxes, ha='right', fontstyle='italic')
        
        # 设置坐标轴标签
        ax.set_xlabel('Safety Rate (%)', fontweight='bold')
        if ax == axes[0]:
            ax.set_ylabel('Success Rate (%)', fontweight='bold')

    # 全局设置
    plt.suptitle('Performance Metrics with 95% Confidence Intervals', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_image, dpi=300, bbox_inches='tight')
    print(f"结果已保存至: {os.path.abspath(output_image)}")

def check_json_mismatches(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):  # 仅处理 JSON 文件
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list) and data:
                        first_mismatch = data[0].get("mismatch_summary", "")
                        if "No mismatches found." in first_mismatch:
                            print(f"Mismatch found in: {filename}")
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error reading {filename}: {e}")

# # 使用示例
# folder_path = "/home/ubuntu/jiajun_benchmark/app-k8s/result/GPT-4o/agent_test/20250322_025008/base"  # 替换为实际路径
# check_json_mismatches(folder_path)


# if "__name__" == "__main__":
#     folder_path = "/home/ubuntu/jiajun_benchmark/app-k8s/result/GPT-4o/20250314_032427"
#     print("1")
#     stat = analyze_and_save_results(folder_path)

    # summary_tests(folder_path)
    # plot_metrics(folder_path)
    # plot_correctness(folder_path)



