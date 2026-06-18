import os
import requests

def download_manual():
    os.makedirs("./data", exist_ok=True)
    file_path = "./data/ur10e_manual.pdf"

    url = "https://s3-eu-west-1.amazonaws.com/ur-support-site/32554/UR10e_User_Manual_en_Global.pdf"

    if os.path.exists(file_path):
        print(f"[INFO] Source file already exists: '{file_path}'. Skipping download.")
        return

    print("[INFO] Downloading UR10e user manual (approx. 10 MB)...")
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        with open(file_path, 'wb') as f:
            f.write(response.content)
        print(f"[INFO] Download complete: '{file_path}'.")
    except Exception as e:
        print(f"[ERROR] Download failed: {e}")

if __name__ == "__main__":
    download_manual()