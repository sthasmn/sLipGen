import sys, pickle
from pathlib import Path

def load_pickle(p):
    with open(p, "rb") as f:
        return pickle.load(f)

def main():
    if len(sys.argv) < 2:
        print("Usage: python inspect_pywork.py <pywork_dir>")
        return
    pywork = Path(sys.argv[1])
    for f in pywork.glob("*.pckl"):
        print(f"\n=== {f.name} ===")
        try:
            obj = load_pickle(f)
            if isinstance(obj, dict):
                print(f"Type: dict, keys: {list(obj.keys())[:10]}")
                first_key = next(iter(obj))
                print(f"Sample [{first_key}]: {obj[first_key]}")
            elif isinstance(obj, list):
                print(f"Type: list, len: {len(obj)}")
                if obj:
                    print("First element:", obj[0])
            else:
                print(f"Type: {type(obj)}")
                print(obj)
        except Exception as e:
            print("Error reading:", e)

if __name__ == "__main__":
    main()
