import argparse
from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser(
        description="Download a HuggingFace dataset snapshot."
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="parameterlab/c-seo-results",
        help="The HuggingFace repo_id for the dataset.",
    )
    parser.add_argument(
        "--local_dir",
        type=str,
        default="../experiments/results",
        help="Local directory to download the dataset into.",
    )
    args = parser.parse_args()

    local_dir = snapshot_download(
        repo_id=args.repo_id, repo_type="dataset", local_dir=args.local_dir
    )
    print(f"Files are now in: {local_dir}")


if __name__ == "__main__":
    main()
