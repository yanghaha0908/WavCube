import os
import argparse
import tqdm


def generate_index(data_path, idx_path):
    """
    生成用于随机读取的偏移量索引文件 (.idx)
    """
    if not os.path.exists(data_path):
        print(f"❌ 错误: 找不到输入文件 {data_path}")
        return

    print(f"正在为 {data_path} 生成索引...")
    print(f"输出路径: {idx_path}")

    offset = 0
    # 注意：输入文件必须用 'rb' (二进制) 读取，以确保字节偏移量准确
    with open(data_path, 'rb') as f_in, open(idx_path, 'w') as f_out:
        for line in tqdm.tqdm(f_in, desc="Processing lines", unit=" lines"):
            f_out.write(f"{offset}\n")
            offset += len(line)

    print(f"✅ 索引生成完毕！")
    print("-" * 30)

    # === 验证环节（只读首尾两行，不把全部 idx 加载进内存）===
    print("正在验证索引准确性...")
    try:
        # 读取第一个偏移量
        with open(idx_path, 'r') as f_idx:
            first_offset = int(f_idx.readline().strip())

        # 读取最后一个偏移量（逐行扫描，内存只保留当前行）
        last_offset = first_offset
        total_lines = 0
        with open(idx_path, 'r') as f_idx:
            for line in f_idx:
                last_offset = int(line.strip())
                total_lines += 1
        print(f"总行数: {total_lines}")

        # 用二进制模式 seek，避免跨平台文本模式问题
        with open(data_path, 'rb') as f_data:
            f_data.seek(first_offset)
            line1 = f_data.readline().decode('utf-8', errors='replace').strip()
            print(f"第 1 行内容: {line1[:100]}")

            f_data.seek(last_offset)
            line_last = f_data.readline().decode('utf-8', errors='replace').strip()
            print(f"最后一行内容: {line_last[:100]}")

    except Exception as e:
        print(f"❌ 验证失败: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="为大文件生成随机读取的偏移量索引 (.idx)")
    parser.add_argument("--input", "-i", required=True, help="输入文件路径")
    parser.add_argument("--output", "-o", default=None, help="输出 .idx 文件路径（默认为 input + .idx）")
    args = parser.parse_args()

    input_file = args.input
    output_idx = args.output if args.output else input_file + ".idx"

    generate_index(input_file, output_idx)
