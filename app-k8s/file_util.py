import json
import os
import itertools
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
    basic_errors = ["remove_ingress", "change_port", "add_egress"]#
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

# 使用示例
folder_path = "/home/ubuntu/jiajun_benchmark/app-k8s/result/GPT-4o/20250305_031344"
plot_metrics(folder_path)
plot_correctness(folder_path)



