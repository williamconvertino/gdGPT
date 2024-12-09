import os
import subprocess

# Define the repository URL (replace with the actual URL of your repository)
REPO_URL = "https://github.com/williamconvertino/llm-research.git"

# Base folder name
base_folder = "RUN_"

# Loop to create folders and clone the repository
for i in range(1, 9):  # 1 through 8
    folder_name = f"{base_folder}{i}"
    if not os.path.exists(folder_name):
        # Create the folder if it doesn't already exist
        os.makedirs(folder_name, exist_ok=True)
        
        # Clone the repository into the folder
        try:
            print(f"Cloning into {folder_name}...")
            subprocess.run(["git", "clone", REPO_URL, folder_name], check=True)
            print(f"Successfully cloned into {folder_name}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to clone repository into {folder_name}: {e}")
    else:
      # Otherwise git stash and pull
      print(f"Pulling into {folder_name}...")
      os.chdir(folder_name)
      subprocess.run(["git", "stash"], check=True)
      subprocess.run(["git", "pull"], check=True)
      os.chdir("..")

print("All operations completed.")
