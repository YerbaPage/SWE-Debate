import json
import os


def convert_to_json(obj):
    if isinstance(obj, list):
        res_obj = []
        for o in obj:
            try:
                json_o = json.loads(o.model_dump_json())
            except:
                json_o = o
            res_obj.append(json_o)
        return res_obj
    else:
        try:
            return json.loads(obj.model_dump_json())
        except:
            print(f'{type(obj)} cannot be converted to json directly.')
            return None


def load_jsonl(filepath):
    """
    Load a JSONL file from the given filepath.

    Arguments:
    filepath -- the path to the JSONL file to load

    Returns:
    A list of dictionaries representing the data in each line of the JSONL file.
    """
    with open(filepath, "r") as file:
        return [json.loads(line) for line in file]


def write_to_jsonl(data, filepath):
    """
    Write data to a JSONL file at the given filepath.

    Arguments:
    data -- a list of dictionaries to write to the JSONL file
    filepath -- the path to the JSONL file to write
    """
    with open(filepath, "w") as file:
        for entry in data:
            file.write(json.dumps(entry) + "\n")


def append_to_jsonl(data, filepath):
    with open(filepath, "a") as file:
        file.write(json.dumps(data) + "\n")
        

def load_json(filepath):
    return json.load(open(filepath, "r"))


def clear_file(filepath):
    with open(filepath, 'w') as f:
        f.write("")
        
        
def backup_file(original_file):
    backup_path = original_file
    if os.path.exists(original_file):
        import shutil
        # Define the backup file path
        backup_path = original_file + '.backup'
        
        # Copy the original file to create a backup
        shutil.copy2(original_file, backup_path)
    return backup_path


def delete_file(file_path):
    try:
        # Delete the file
        os.remove(file_path)
        # print(f"File '{file_path}' has been deleted successfully.")
    except FileNotFoundError:
        print(f"File '{file_path}' does not exist.")
    except PermissionError:
        print(f"Permission denied: Cannot delete '{file_path}'.")
    except Exception as e:
        print(f"An error occurred while deleting the file: {e}")