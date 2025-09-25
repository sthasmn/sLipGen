import os
import shutil

root = "/mnt/e/JCOM/RAW_Data/ふわっと欣様"   # 🔧 change to your real path
keep = {"output"}        # ✅ folders with these names will not be deleted

for item in os.listdir(root):
    top_path = os.path.join(root, item)
    if os.path.isdir(top_path):  # Only process folders in root
        for sub in os.listdir(top_path):
            sub_path = os.path.join(top_path, sub)
            if os.path.isdir(sub_path) and sub not in keep:
                print(f"Deleting folder: {sub_path}")
                shutil.rmtree(sub_path)  # Recursively delete subfolder
