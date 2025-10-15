from huggingface_hub import upload_folder

upload_folder(
    folder_path="models/best",  # path to your model folder
    repo_id="MelvynCHEMIN/query_classifier",
    repo_type="model")