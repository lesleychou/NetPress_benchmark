import os
import json
import matplotlib.pyplot as plt
import shutil

def prepare_file(file_path):
    """
    Prepares the specified file for use. If it exists, clears its content.
    If it doesn't exist, creates the file and any missing directories.

    Parameters:
        file_path (str): The path to the file.
    """
    if os.path.exists(file_path):
        # Clear the file's content
        with open(file_path, "w") as f:
            pass
    else:
        # Create missing directories and the file
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            f.write("")

def delete_result_folder(folder_path):
    """
    Deletes the specified folder and its contents if it exists.

    Parameters:
        folder_path (str): The path to the folder.
    """
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)

def initialize_json_file(json_path):
    """
    Create an empty JSON file and its directory if they don't exist.
    
    Args:
        json_path (str): The path of the JSON file to initialize.
    """
    # Ensure the directory exists
    directory = os.path.dirname(json_path)
    if directory:  # Check if the path includes a directory
        os.makedirs(directory, exist_ok=True)

    # Create the JSON file if it doesn't exist
    if not os.path.exists(json_path):
        with open(json_path, 'w') as json_file:
            json.dump([], json_file, indent=4)

def summarize_results(json_folder, output_file):
    """Summarize results from multiple JSON files in a folder and write to a new JSON file."""
    total_success = 0
    total_iterations = 0
    total_average_time = 0
    total_safe_files = 0
    total_packet_loss_reduction = 0
    total_files_with_reduction = 0

    # Initialize error-specific statistics
    router_errors_count = 0
    router_errors_success = 0
    routing_table_errors_count = 0
    routing_table_errors_success = 0

    # Get all JSON file paths in the folder
    json_files = [
        os.path.join(json_folder, f)
        for f in os.listdir(json_folder)
        if f.endswith(".json") and os.path.isfile(os.path.join(json_folder, f))
    ]
    total_files = len(json_files)

    if total_files == 0:
        raise ValueError("No valid JSON files found in the specified folder.")

    for json_path in json_files:
        with open(json_path, "r") as file:
            data = json.load(file)

        # Analyze the second-to-last entry for success
        second_last_entry = data[-2]
        last_entry = data[-1]
        success = 1 if second_last_entry.get("packet_loss", -1) == 0 else 0

        # Count total successes
        total_success += success

        # Check for error type and update statistics
        if last_entry.get("routercheck") == 1:
            router_errors_count += 1
            router_errors_success += success
        if last_entry.get("routingtablecheck") == 1:
            routing_table_errors_count += 1
            routing_table_errors_success += success

        # Calculate average elapsed time and iterations
        elapsed_times = [entry["elapsed_time"] for entry in data if "elapsed_time" in entry]
        average_time = sum(elapsed_times) / len(elapsed_times)

        # Check if the file is safe (packet_loss is non-increasing)
        packet_losses = [entry["packet_loss"] for entry in data if "packet_loss" in entry]
        is_safe = all(x >= y for x, y in zip(packet_losses, packet_losses[1:]))
        total_safe_files += 1 if is_safe else 0

        # Calculate packet loss reduction
        if len(packet_losses) >= 2:
            initial_packet_loss = packet_losses[0]
            final_packet_loss = packet_losses[-1]
            packet_loss_reduction = initial_packet_loss - final_packet_loss
            total_packet_loss_reduction += packet_loss_reduction
            total_files_with_reduction += 1

        total_average_time += average_time
        total_iterations += len(elapsed_times)

    # Calculate overall metrics
    success_rate = total_success / total_files
    overall_average_time = total_average_time / total_files
    average_iterations = total_iterations / total_files
    safety_rate = total_safe_files / total_files

    # Calculate success rates for each error type
    router_errors_success_rate = (
        router_errors_success / router_errors_count if router_errors_count > 0 else 0
    )
    routing_table_errors_success_rate = (
        routing_table_errors_success / routing_table_errors_count if routing_table_errors_count > 0 else 0
    )

    # Calculate average packet loss reduction
    average_packet_loss_reduction = (
        total_packet_loss_reduction / total_files_with_reduction
        if total_files_with_reduction > 0 else 0
    )

    # Prepare the summary
    summary = {
        "success_rate": round(success_rate, 2),
        "overall_average_time": round(overall_average_time, 2),
        "average_iterations": round(average_iterations, 2),
        "safety_rate": round(safety_rate, 2),
        "average_packet_loss_reduction": round(average_packet_loss_reduction, 2),
        "error_statistics": {
            "routercheck": {
                "count": router_errors_count,
                "success_rate": round(router_errors_success_rate, 2),
            },
            "routingtablecheck": {
                "count": routing_table_errors_count,
                "success_rate": round(routing_table_errors_success_rate, 2),
            }
        }
    }

    # Write the summary to the output file
    with open(output_file, "w") as out_file:
        json.dump(summary, out_file, indent=4)

def error_classification(errors, json_path):
    """
    Classify errors into router and routing table errors and update the JSON file with the classification.

    Parameters:
        errors (list): List of error functions.
        json_path (str): Path to the JSON file to update.
    """
    # Define error categories
    router_errors = {
        "error_disable_routing",
        "error_disable_interface",
        "error_drop_traffic_to_from_subnet"
    }
    routing_table_errors = {
        "error_remove_ip",
        "error_wrong_routing_table"
    }

    # Extract the names of the errors
    error_names = {error.__name__ for error in errors}

    # Determine the values for routercheck and routingtablecheck
    routercheck = 1 if error_names & router_errors else 0
    routingtablecheck = 1 if error_names & routing_table_errors else 0

    # Read the JSON file
    with open(json_path, 'r') as file:
        data = json.load(file)

    # Append new fields to the data
    data.append({
        "routercheck": routercheck,
        "routingtablecheck": routingtablecheck
    })

    # Write the updated data back to the output JSON file
    with open(json_path, 'w') as file:
        json.dump(data, file, indent=4)

def plot_metrics_from_json(json_path, output_image_path):
    """
    Plot success rates and other metrics from a JSON file and save to an image file.

    Parameters:
        json_path (str): Path to the JSON file containing the metrics.
        output_image_path (str): Path to save the output image.
    """
    # Load data from JSON file
    if not os.path.isfile(json_path):
        raise ValueError("Invalid JSON file path.")

    with open(json_path, "r") as file:
        data = json.load(file)

    # Extract success rates
    success_rate = data.get("success_rate", 0) * 100
    routercheck_success_rate = data.get("error_statistics", {}).get("routercheck", {}).get("success_rate", 0) * 100
    routingtablecheck_success_rate = data.get("error_statistics", {}).get("routingtablecheck", {}).get("success_rate", 0) * 100

    # Extract other metrics
    overall_average_time = data.get("overall_average_time", 0)
    average_iterations = data.get("average_iterations", 0)
    average_packet_loss_reduction = data.get("average_packet_loss_reduction", 0)

    # Plot success rates
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # First plot: Success rates
    labels = ["Overall Success Rate", "Routercheck Success Rate", "Routingtablecheck Success Rate"]
    values = [success_rate, routercheck_success_rate, routingtablecheck_success_rate]
    ax[0].bar(labels, values, color=["blue", "orange", "green"])
    ax[0].set_title("Success Rates")
    ax[0].set_ylabel("Rate (%)")
    ax[0].set_ylim(0, max(values) + 5)
    ax[0].grid(axis="y", linestyle="--", alpha=0.7)
    ax[0].set_xticks(range(len(labels)))
    ax[0].set_xticklabels(labels, rotation=45, ha="right", fontsize=10)

    # Second plot: Other metrics
    metrics = ["Overall Avg Time (s)", "Avg Iterations (count)", "Avg Packet Loss Reduction (%)"]
    values = [overall_average_time, average_iterations, average_packet_loss_reduction]
    ax[1].bar(metrics, values, color=["blue", "orange", "red"])
    ax[1].set_title("Other Metrics")
    ax[1].set_ylabel("Value")
    ax[1].grid(axis="y", linestyle="--", alpha=0.7)
    ax[1].set_xticks(range(len(metrics)))
    ax[1].set_xticklabels(metrics, rotation=45, ha="right", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_image_path)
    plt.close()

def plot_metrics(result_dir, error_types):
    """
    Plot success rates, safety rates, and average iterations for different error types.

    Parameters:
        result_dir (str): Directory containing the result JSON files.
        error_types (list): List of error types to plot.
    """
    success_rates = []
    safety_rates = []
    average_iterations = []

    for error_type in error_types:
        json_path = os.path.join(result_dir, error_type, f'{error_type}_result.json')
        with open(json_path, 'r') as f:
            data = json.load(f)
            success_rates.append(data.get('success_rate', 0))
            safety_rates.append(data.get('safety_rate', 0))
            average_iterations.append(data.get('average_iterations', 0))

    # Plot success rates
    plt.figure(figsize=(12, 6))
    plt.bar(error_types, success_rates, color='blue')
    plt.xlabel('Error Type')
    plt.ylabel('Success Rate')
    plt.title('Success Rate by Error Type')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'success_rate.png'))
    plt.close()

    # Plot safety rates
    plt.figure(figsize=(12, 6))
    plt.bar(error_types, safety_rates, color='green')
    plt.xlabel('Error Type')
    plt.ylabel('Safety Rate')
    plt.title('Safety Rate by Error Type')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'safety_rate.png'))
    plt.close()

    # Plot average iterations
    plt.figure(figsize=(12, 6))
    plt.bar(error_types, average_iterations, color='red')
    plt.xlabel('Error Type')
    plt.ylabel('Average Iterations')
    plt.title('Average Iterations by Error Type')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'average_iterations.png'))
    plt.close()

def plot_combined_error_metrics(result_dir, error_combinations):
    """
    Plot success rates, safety rates, and average iterations for combined error types.

    Parameters:
        result_dir (str): Directory containing the result JSON files.
        error_combinations (list): List of tuples containing combined error types.
    """
    success_rates = []
    safety_rates = []
    average_iterations = []

    for i, (error1, error2) in enumerate(error_combinations):
        json_path = os.path.join(result_dir, f'test_{i+1}', f'result_{i+1}.json')
        with open(json_path, 'r') as f:
            data = json.load(f)
            
            # Ignore the last entry
            data = data[:-1]

            
            success_rate = sum(1 for entry in data if entry.get('packet_loss', 0) == 0) / len(data)
            safety_rate = all(data[i]['packet_loss'] <= data[i-1]['packet_loss'] for i in range(1, len(data)))
            average_iteration = len(data)
            success_rates.append(success_rate)
            safety_rates.append(1 if safety_rate else 0)
            average_iterations.append(average_iteration)

    labels = [f'{error1} + {error2}' for error1, error2 in error_combinations]

    # Plot success rates
    plt.figure(figsize=(12, 6))
    plt.bar(labels, success_rates, color='blue')
    plt.xlabel('Error Combinations')
    plt.ylabel('Success Rate')
    plt.title('Success Rate by Error Combinations')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'success_rate.png'))
    plt.close()

    # Plot safety rates
    plt.figure(figsize=(12, 6))
    plt.bar(labels, safety_rates, color='green')
    plt.xlabel('Error Combinations')
    plt.ylabel('Safety Rate')
    plt.title('Safety Rate by Error Combinations')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'safety_rate.png'))
    plt.close()

    # Plot average iterations
    plt.figure(figsize=(12, 6))
    plt.bar(labels, average_iterations, color='red')
    plt.xlabel('Error Combinations')
    plt.ylabel('Average Iterations')
    plt.title('Average Iterations by Error Combinations')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'average_iterations.png'))
    plt.close()

