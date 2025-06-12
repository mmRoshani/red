import os
import shutil
import random
from random import choice, sample, shuffle
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=str, required=True, help='Input base directory containing class subdirectories')
    parser.add_argument('--output', type=str, required=True, help='Relative path for output client directories')
    parser.add_argument('--input2', type=str, help='Second input base directory (for balanced splitting)')
    parser.add_argument('--output2', type=str, help='Output directory for second input splitting')
    parser.add_argument('--max', type=int, default=10, help='Maximum samples per class')
    return parser.parse_args()

def balanced_split_all_classes(input_dir2, output_dir2, labels, num_clients):
    print(f"\nPerforming balanced split of second dataset into {num_clients} clients...")
    os.makedirs(output_dir2, exist_ok=True)

    for i in range(1, num_clients + 1):
        for label in labels:
            os.makedirs(os.path.join(output_dir2, f'client_{i}', label), exist_ok=True)

    for label in labels:
        class_dir = os.path.join(input_dir2, label)
        txt_files = [f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f)) and f.lower().endswith('.txt')]

        if not txt_files:
            print(f"No .txt files found in class '{label}', skipping.")
            continue

        random.shuffle(txt_files)
        total = len(txt_files)
        base_count = total // num_clients
        remainder = total % num_clients

        start = 0
        for i in range(num_clients):
            count = base_count + (1 if i < remainder else 0)
            client_dir = os.path.join(output_dir2, f'client_{i+1}', label)
            selected = txt_files[start:start+count]
            for fname in selected:
                shutil.copy2(os.path.join(class_dir, fname), os.path.join(client_dir, fname))
            print(f"Copied {len(selected)} '{label}' files to client_{i+1}/{label}")
            start += count

    print("Second balanced splitting complete.")

def main():
    seed = 42
    random.seed(seed)
    args = parse_args()

    input_dir = os.path.abspath(args.directory)
    output_base = os.path.abspath(os.path.join(args.directory, args.output))  # Resolve relative output path

    labels = [name for name in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, name))]
    labels.sort()
    num_labels = len(labels)

    if num_labels < 2:
        print("Need at least two labels (subdirectories) to assign classes.")
        return

    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}

    caps = [args.max] * num_labels
    assignments = []

    while sum(caps) >= 2:
        first = choice([i for i, v in enumerate(caps) if v > 0])
        caps[first] -= 1
        avail = [i for i, v in enumerate(caps) if v > 0 and i != first]
        second = choice(avail) if avail else first
        caps[second] -= 1
        assignments.append((first, second))

    print("Label mapping:")
    for idx, label in idx_to_label.items():
        print(f" {idx+1}: {label}")

    print(f"\nAssigned to {len(assignments)} clients:")
    for i, (a, b) in enumerate(assignments, 1):
        print(f" Client #{i}: {idx_to_label[a]} & {idx_to_label[b]}")

    class_map = {idx: [] for idx in range(num_labels)}
    for client_idx, (a, b) in enumerate(assignments, 1):
        class_map[a].append(client_idx)
        class_map[b].append(client_idx)

    print("\nCreating client directories and label subfolders...")
    for client_idx in range(1, len(assignments)+1):
        client_dir = os.path.join(output_base, f"client_{client_idx}")
        for label in labels:
            os.makedirs(os.path.join(client_dir, label), exist_ok=True)

    print("\nSplitting .txt files per class among clients...")
    for class_idx, clients in class_map.items():
        if not clients:
            continue

        label = idx_to_label[class_idx]
        src_dir = os.path.join(input_dir, label)
        txt_files = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f)) and f.lower().endswith('.txt')]

        if not txt_files:
            print(f" No .txt files found in class '{label}', skipping.")
            continue

        shuffle(txt_files)

        k = len(clients)
        if k == 1:
            percents = [100]
        else:
            cuts = sorted(sample(range(1, 100), k - 1))
            cuts = [0] + cuts + [100]
            percents = [cuts[i + 1] - cuts[i] for i in range(k)]

        total = len(txt_files)
        counts = [int((p / 100) * total) for p in percents]
        remainder = total - sum(counts)
        counts[-1] += remainder

        start = 0
        for client_pos, client_idx in enumerate(clients):
            count = counts[client_pos]
            selected = txt_files[start:start + count]
            start += count

            dest_dir = os.path.join(output_base, f"client_{client_idx}", label)

            for fname in selected:
                shutil.copy2(
                    os.path.join(src_dir, fname),
                    os.path.join(dest_dir, fname)
                )

            print(f"Copied {len(selected)} files of '{label}' to client_{client_idx}/{label}/")

    num_clients = len(assignments)
    if args.input2 and args.output2:
        input2_dir = os.path.abspath(args.input2)
        output2_dir = os.path.abspath(args.output2)

        balanced_split_all_classes(input2_dir, output2_dir, labels, num_clients)


    print("\nData splitting complete.")

if __name__ == "__main__":
    main()
