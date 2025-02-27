import json

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
