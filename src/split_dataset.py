import argparse, os, shutil, random


def copy_split(src_label_dir, dst_train, dst_val, val_ratio):
    files = [
        f
        for f in os.listdir(src_label_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    random.shuffle(files)
    n_val = int(len(files) * val_ratio)
    val_files = set(files[:n_val])
    label = os.path.basename(src_label_dir)
    os.makedirs(os.path.join(dst_train, label), exist_ok=True)
    os.makedirs(os.path.join(dst_val, label), exist_ok=True)
    for f in files:
        src = os.path.join(src_label_dir, f)
        dst_root = dst_val if f in val_files else dst_train
        shutil.copy2(src, os.path.join(dst_root, label, f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--val_ratio", type=float, default=0.2)
    args = ap.parse_args()
    random.seed(42)

    for label in os.listdir(args.raw):
        p = os.path.join(args.raw, label)
        if os.path.isdir(p):
            copy_split(p, args.train, args.val, args.val_ratio)
    print("Split complete.")


if __name__ == "__main__":
    main()
