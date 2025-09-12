import os

def print_tree(folder_path, prefix=""):
    # List all items in the folder
    items = os.listdir(folder_path)
    for i, item in enumerate(items):
        item_path = os.path.join(folder_path, item)
        # Decide the branch symbol
        branch = "├── " if i < len(items) - 1 else "└── "
        print(prefix + branch + item)
        # Recurse into subfolders
        if os.path.isdir(item_path):
            extension = "│   " if i < len(items) - 1 else "    "
            print_tree(item_path, prefix + extension)

# Example usage
folder_to_draw = "."
print_tree(folder_to_draw)
