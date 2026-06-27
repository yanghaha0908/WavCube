import os
import argparse
import tqdm


def generate_index(data_path, idx_path):
    """
    Generate an offset index file (.idx) for random reads
    """
    if not os.path.exists(data_path):
        print(f"❌ Error: input file not found {data_path}")
        return

    print(f"Generating index for {data_path}...")
    print(f"Output path: {idx_path}")

    offset = 0
    # Note: the input file must be read in 'rb' (binary) mode to ensure byte offsets are accurate
    with open(data_path, 'rb') as f_in, open(idx_path, 'w') as f_out:
        for line in tqdm.tqdm(f_in, desc="Processing lines", unit=" lines"):
            f_out.write(f"{offset}\n")
            offset += len(line)

    print(f"✅ Index generation complete!")
    print("-" * 30)

    # === Verification step (only reads the first and last lines, does not load the entire idx into memory) ===
    print("Verifying index accuracy...")
    try:
        # Read the first offset
        with open(idx_path, 'r') as f_idx:
            first_offset = int(f_idx.readline().strip())

        # Read the last offset (scan line by line, keeping only the current line in memory)
        last_offset = first_offset
        total_lines = 0
        with open(idx_path, 'r') as f_idx:
            for line in f_idx:
                last_offset = int(line.strip())
                total_lines += 1
        print(f"Total lines: {total_lines}")

        # Use binary mode seek to avoid cross-platform text mode issues
        with open(data_path, 'rb') as f_data:
            f_data.seek(first_offset)
            line1 = f_data.readline().decode('utf-8', errors='replace').strip()
            print(f"Line 1 content: {line1[:100]}")

            f_data.seek(last_offset)
            line_last = f_data.readline().decode('utf-8', errors='replace').strip()
            print(f"Last line content: {line_last[:100]}")

    except Exception as e:
        print(f"❌ Verification failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate an offset index (.idx) for random reads of large files")
    parser.add_argument("--input", "-i", required=True, help="Input file path")
    parser.add_argument("--output", "-o", default=None, help="Output .idx file path (defaults to input + .idx)")
    args = parser.parse_args()

    input_file = args.input
    output_idx = args.output if args.output else input_file + ".idx"

    generate_index(input_file, output_idx)
