from huggingface_hub import snapshot_download

# Downloads everything in the repo to a local directory
repo_path = snapshot_download(repo_id="MMVP/MMVP_VLM", repo_type="dataset")

print("Dataset downloaded to:", repo_path)

