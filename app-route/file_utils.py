import os
import json

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

import os
import json

import os
import json

def summarize_results(json_folder, output_file):
    """Summarize results from multiple JSON files in a folder and write to a new JSON file."""
    total_success = 0
    total_iterations = 0
    total_average_time = 0
    total_safe_files = 0

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

        last_entry = data[-1]
        total_success += 1 if last_entry["packet_loss"] == 0 else 0
        elapsed_times = [entry["elapsed_time"] for entry in data]
        average_time = sum(elapsed_times) / len(elapsed_times)

        # Check if the file is safe (packet_loss is non-increasing)
        packet_losses = [entry["packet_loss"] for entry in data]
        is_safe = all(x >= y for x, y in zip(packet_losses, packet_losses[1:]))
        total_safe_files += 1 if is_safe else 0

        total_average_time += average_time
        total_iterations += len(elapsed_times)

    success_rate = total_success / total_files
    overall_average_time = total_average_time / total_files
    average_iterations = total_iterations / total_files
    safety_rate = total_safe_files / total_files

    summary = {
        "success_rate": round(success_rate, 2),
        "overall_average_time": round(overall_average_time, 2),
        "average_iterations": round(average_iterations, 2),
        "safety_rate": round(safety_rate, 2)
    }

    with open(output_file, "w") as out_file:
        json.dump(summary, out_file, indent=4)






