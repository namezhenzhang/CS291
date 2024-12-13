import os
import subprocess
import json
from datasets import load_dataset

def clone_and_extract_files(repo_path, commit_hash):
    """
    Clone a Git repository if not already present, update to the latest version if present,
    checkout a specific commit, and extract all files into a list of dictionaries.

    Args:
        repo_path (str): The GitHub repository path, e.g., "DataDog/integrations-core".
        commit_hash (str): The commit hash to checkout.

    Returns:
        list: A list of dictionaries where each dictionary contains {file_path: file_content}.
    """

    repo_url = f"https://github.com/{repo_path}.git"
    repo_name = repo_path.split('/')[-1]
    temp_dir = os.path.join( os.getcwd(), "repos", repo_name)

    if not os.path.exists(temp_dir):
        print(f"Cloning repository {repo_url} into {temp_dir}...")
        subprocess.run(["git", "clone", repo_url, temp_dir], check=True)
    else:
        print(f"Repository already exists at {temp_dir}.")

    os.chdir(temp_dir)
    print(f"Checking out commit {commit_hash}...")
    subprocess.run(["git", "checkout", commit_hash], check=True)

    files_list = []
    for root, _, files in os.walk(temp_dir):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            relative_path = os.path.relpath(file_path, temp_dir)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    file_content = f.read()
                files_list.append((relative_path, file_content))
            except Exception as e:
                # print(f"Error reading file {relative_path}: {e}")
                files_list.append((relative_path, None))
    os.chdir("../..")

    return files_list

if __name__ == "__main__":
    ds = load_dataset("princeton-nlp/SWE-bench")
    train_dataset = ds['train']
    repo = "DataDog/integrations-core"
    commit = "160cfef6e1118061fa66d333e8c2a572f5d0a815"
    result = clone_and_extract_files(repo, commit)
    output_path = f"./pythonProject/{commit}.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(result, output_file, ensure_ascii=False, indent=4)
    print("Done")