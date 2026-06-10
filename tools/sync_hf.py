import argparse
import os
from huggingface_hub import HfApi, snapshot_download


def main():
    parser = argparse.ArgumentParser(description="Sync artifacts with Hugging Face Hub")
    parser.add_argument(
        "--repo-id", type=str, required=True, help="Hugging Face repository ID"
    )
    parser.add_argument(
        "--local-dir", type=str, required=True, help="Local directory to sync"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face API token (optional if already logged in)",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--upload", action="store_true", help="Upload local directory to Hugging Face"
    )
    group.add_argument(
        "--download",
        action="store_true",
        help="Download from Hugging Face to local directory",
    )

    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")

    if args.upload:
        print(f"Uploading {args.local_dir} to {args.repo_id}...")
        api = HfApi(token=token)
        try:
            api.create_repo(repo_id=args.repo_id, exist_ok=True)
        except Exception as e:
            print(
                f"Note: Could not create repo (it might already exist or token lacks permission): {e}"
            )

        api.upload_folder(
            folder_path=args.local_dir,
            repo_id=args.repo_id,
            repo_type="model",
        )
        print("Upload complete!")

    elif args.download:
        print(f"Downloading from {args.repo_id} to {args.local_dir}...")
        snapshot_download(
            repo_id=args.repo_id,
            local_dir=args.local_dir,
            token=token,
            repo_type="model",
        )
        print("Download complete!")


if __name__ == "__main__":
    main()
