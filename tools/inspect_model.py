import sys
from pathlib import Path
from ultralytics import YOLO

def inspect_mystery_pt(file_path):
    print("=========================================")
    print(f" Inspecting: {file_path}")
    print("=========================================\n")
    
    try:
        # Load the model
        model = YOLO(file_path)
        ckpt = model.ckpt
        
        print(f"[+] Task Type: {model.task}")

        # Date and Version
        raw_date = ckpt.get('date', 'Not recorded')
        print(f"[+] Date Trained/Saved: {raw_date}")
        print(f"[+] Ultralytics Version: {ckpt.get('version', 'Not recorded')}")
        
        # Classes
        classes = model.names
        print(f"\n[+] Total Classes: {len(classes)}")
        print(f"[+] Class Dictionary: {classes}")

        # Training Progress
        epoch = ckpt.get('epoch', -1)
        if epoch != -1:
            print(f"[+] Stopped at Epoch: {epoch}")
        
        # Training Arguments (The exact settings used to train it)
        if 'train_args' in ckpt:
            args = ckpt['train_args']
            print("\n[+] Training Settings:")
            print(f"  [+] Dataset Configuration File: {args.get('data', 'Unknown')}")
            print(f"  [+] Image Resolution (imgsz): {args.get('imgsz', 'Unknown')}")
            print(f"  [+] Target Epochs: {args.get('epochs', 'Unknown')}")
            print(f"  [+] Batch Size: {args.get('batch', 'Unknown')}")
            print(f"  [+] Optimizer Used: {args.get('optimizer', 'Unknown')}")
            print(f"  [+] Initial Learning Rate: {args.get('lr0', 'Unknown')}")
        else:
            print("\n[-] No training arguments found in this file.")

        print("\n[+] Architecture Summary:")
        model.info()

        # Generate Suggested Filename
        # Format: YYMMDD_[sim|real]_[Arch]_[Task]_[Classes]cls.pt
        if isinstance(raw_date, str) and len(raw_date) >= 10:
            yymmdd = raw_date[2:4] + raw_date[5:7] + raw_date[8:10]
        else:
            yymmdd = "YYMMDD"
            
        task_name = model.task if hasattr(model, 'task') else "task"
        class_count = len(classes)
        
        suggested_name = f"{yymmdd}_[sim|real]_[Arch]_{task_name}_{class_count}cls.pt"
        
        print(f"\n[+] Suggested Filename:")
        print(f"    -> {suggested_name}")

    except Exception as e:
        print(f"Failed to load via Ultralytics: {e}")
        print("This might not be a standard Ultralytics YOLO model, or it requires a custom repository.")


if __name__ == "__main__":
    # Ensure the user provided an argument
    if len(sys.argv) < 2:
        input_path = Path("models")
        # print("Usage: python inspect_model.py <path_to_file_or_folder>")
        # sys.exit(1)
    else:
        input_path = Path(sys.argv[1])

    # Check if the path exists
    if not input_path.exists():
        print(f"[-] Error: The path '{input_path}' does not exist.")
        sys.exit(1)

    # If it's a single file
    if input_path.is_file():
        if input_path.suffix == '.pt':
            inspect_mystery_pt(str(input_path))
        else:
            print(f"[-] Error: '{input_path}' is not a .pt file.")
            
    # If it's a folder
    elif input_path.is_dir():
        # Find all .pt files in the directory
        pt_files = list(input_path.glob("*.pt"))
        
        if not pt_files:
            print(f"[-] No .pt files found in folder: '{input_path}'")
        else:
            print(f"[+] Found {len(pt_files)} model(s) in '{input_path}'. Starting inspection...\n")
            for pt_file in pt_files:
                inspect_mystery_pt(str(pt_file))