import os
import json
import re
import pandas as pd
import argparse
from pathlib import Path

def extract_info_from_foldername(folder_name):
    """同时提取 step 和 val_loss"""
    step_match = re.search(r'step=(\d+)', folder_name)
    loss_match = re.search(r'val_loss=([0-9.]+)', folder_name)
    
    step = int(step_match.group(1)) if step_match else -1
    loss = float(loss_match.group(1)) if loss_match else -1.0
    
    return step, loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True, help="Outputs root Path")
    args = parser.parse_args()
    
    OUTPUTS_ROOT = args.root

    root_path = Path(OUTPUTS_ROOT)
    data_list = []
    # 遍历目录下所有文件夹
    for folder in root_path.iterdir():
        if folder.is_dir():
            json_file = folder / "_metric2.json"
            with open(json_file, 'r', encoding='utf-8') as f:
                metrics = json.load(f)
            
            step, val_loss = extract_info_from_foldername(folder.name)
            row = {
                "Step": step,
                "Val Loss": val_loss,
                "STOI": metrics.get("stoi", [0])[0],
                "PESQ NB": metrics.get("pesq_nb", [0])[0],
                "PESQ WB": metrics.get("pesq_wb", [0])[0],
                "UTMOS": metrics.get("utmos", [0])[0],
                "SIM": metrics.get("SIM", [0])[0],
                "WER": metrics.get("wer", [0])[0]
            }
            data_list.append(row)


    df = pd.DataFrame(data_list)
    df = df.sort_values(by="Step")
    
    # 调整列顺序
    cols = ["Step", "Val Loss", "STOI", "PESQ NB", "PESQ WB", "UTMOS", "SIM", "WER"]
    df = df[cols]

    output_file = os.path.join(OUTPUTS_ROOT, "final_metrics.csv")
    df.to_csv(output_file, index=False, float_format='%.4f')

    print(f"\n✅ 成功！表格已保存为文件: {output_file}")
    
    print("\n=== 结果预览 ===")
    print(df.to_string(index=False, formatters={c: "{:.4f}".format for c in cols if c != "Step"}))

if __name__ == "__main__":
    main()