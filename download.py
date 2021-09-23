# Source: https://stackoverflow.com/a/39225039
import os
import requests


def download_file_from_google_drive(id, destination):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768
        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()
    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id,
                  'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


if __name__ == "__main__":
    os.makedirs('./checkpoints', exist_ok=True)
    os.makedirs('./examples', exist_ok=True)

    if not os.path.exists('./checkpoints/nyu.pt'):
        print('downloading the model trained on NYUv2...')
        download_file_from_google_drive('1RNiYw5rrqgBf3OkFSCSSQ67s0HMBpkAv', './checkpoints/nyu.pt')

    if not os.path.exists('./checkpoints/scannet.pt'):
        print('downloading the model trained on ScanNet...')
        download_file_from_google_drive('1lOgY9sbMRW73qNdJze9bPkM2cmfA8Re-', './checkpoints/scannet.pt')

    if not os.path.exists('./examples/examples.zip'):
        print('downloading test images...')
        download_file_from_google_drive('1bGZ4VFGkqrTLzQs0ELxEKo8xe_1Sfejg', './examples/examples.zip')
