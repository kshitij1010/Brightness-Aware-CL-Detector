import os

def validate_trial_file(trial_file_path, base_folder):
    """
    Validate the trial file by checking whether all listed files exist in the specified base folder.

    Parameters:
        trial_file_path (str): Path to the trial file.
        base_folder (str): Path to the base folder where files should exist.
    
    Returns:
        None
    """
    try:
        # Read the trial file
        with open(trial_file_path, 'r') as f:
            trial_entries = f.readlines()
        
        missing_files = []
        
        # Check each file in the trial file
        for entry in trial_entries:
            entry = entry.strip()  # Remove any trailing newline characters
            user_step_file = entry.split(':')[1]  # Extract file path after user:step
            full_path = os.path.join(base_folder, user_step_file)

            if not os.path.isfile(full_path):
                missing_files.append(full_path)

        # Report results
        if missing_files:
            print("[ERROR] Missing files:")
            for file in missing_files:
                print(f" - {file}")
            print("\nPlease check your trial file or the directory structure.")
        else:
            print("[SUCCESS] All files in the trial file exist.")
    
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")

# Specify the paths
trial_file_path = "data/trial_file.txt"  # Update this if the trial file is in a different location
base_folder = "data/processed/right"    # Base folder where user directories are located

# Run the validation
validate_trial_file(trial_file_path, base_folder)