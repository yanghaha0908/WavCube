"""
将 logs/ 文件夹上传到 HuggingFace: yhaha/WavCube

用法:
    python upload_to_hf.py
"""

from huggingface_hub import HfApi, login


REPO_ID    = "yhaha/WavCube"
REPO_TYPE  = "model"
LOCAL_DIR  = "logs"
HF_TOKEN   = "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"


def main():
    login(token=HF_TOKEN)

    api = HfApi()

    # 确保 repo 存在，不存在则自动创建
    api.create_repo(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        exist_ok=True,
    )
    print(f"[INFO] Repo: https://huggingface.co/{REPO_ID}")

    # upload_large_folder: 专为大文件 / 大批量设计
    #   - 自动分片上传（chunk upload）
    #   - 断点续传：中断后重跑会跳过已上传的文件
    #   - 多线程并发
    print(f"[INFO] Uploading '{LOCAL_DIR}/' -> repo root ...")
    api.upload_large_folder(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        folder_path=LOCAL_DIR,
    )
    print("[INFO] Done.")
    print(f"       https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
