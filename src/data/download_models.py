import os
import requests
import zipfile

import sys
import shutil


def download_language_models(url):
    current_dir = os.getcwd()
    model_dir = os.path.join(os.path.dirname(os.path.dirname(current_dir)), "models")
    response = requests.get(url)
    if response.status_code == 200:
        with open("lms.zip", "wb") as f:
            f.write(response.content)
        print("File downloaded successfully.")
        print(f"unzipping in {model_dir}")
        with zipfile.ZipFile("lms.zip", "r") as zip_ref:
            zip_ref.extractall(model_dir)
        os.remove('lms.zip')
        if os.path.exists(os.path.join(model_dir, '__MACOSX')):
            shutil.rmtree(os.path.join(model_dir, '__MACOSX'))
    else:
        print("Failed to download file.")


if __name__ == "__main__":
    url = sys.argv[1]
    download_language_models(url)
