import os
import shutil

def clean_output_folders(input_path):
    """
    Traverse all subfolders of input_path, find 'output' folders,
    and delete them if they contain ONLY manifest.json (and no mp4/txt).
    """
    for root, dirs, files in os.walk(input_path):
        # Only check folders that are exactly named 'output'
        for d in dirs:
            if d == "output":
                output_path = os.path.join(root, d)

                # List files inside 'output'
                files_inside = os.listdir(output_path)
                mp4_files = [f for f in files_inside if f.lower().endswith(".mp4")]
                txt_files = [f for f in files_inside if f.lower().endswith(".txt")]
                manifest_exists = "manifest.json" in files_inside

                # Condition: manifest exists, but NO mp4 and NO txt
                if manifest_exists and not mp4_files and not txt_files:
                    print(f"Deleting folder: {output_path}")
                    shutil.rmtree(output_path)  # Permanently delete
                else:
                    print(f"Skipping folder: {output_path} "
                          f"(mp4:{len(mp4_files)}, txt:{len(txt_files)}, manifest:{manifest_exists})")

if __name__ == "__main__":
    # 🔧 Replace with your path
    input_path = "/mnt/e/JCOM/RAW_Data/ふわっと欣様"
    clean_output_folders(input_path)
