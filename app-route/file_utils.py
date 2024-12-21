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

def summarize_results(json_folder, output_file):
    """Summarize results from multiple JSON files in a folder and write to a new JSON file."""
    total_success = 0
    total_iterations = 0
    total_average_time = 0
    total_safe_files = 0

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

        # Analyze the last entry for error type and success
        last_entry = data[-1]
        success = 1 if last_entry.get("packet_loss", -1) == 0 else 0

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

    # Prepare the summary
    summary = {
        "success_rate": round(success_rate, 2),
        "overall_average_time": round(overall_average_time, 2),
        "average_iterations": round(average_iterations, 2),
        "safety_rate": round(safety_rate, 2),
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







