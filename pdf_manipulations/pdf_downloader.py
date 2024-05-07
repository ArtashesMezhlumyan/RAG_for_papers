import os
import requests
import time
import pandas as pd


def download_pdf(url: str, id: str) -> str:
    file_path = f'{id}.pdf'
    response = requests.get(url)
    if response.status_code == 200:
        with open(file_path, 'wb') as file:
            file.write(response.content)
        return f'PDF downloaded successfully: {file_path}'
    else:
        return 'Failed to download the PDF. Status code:', response.status_code


def check_and_download_pdf(df):
    os.makedirs('pdf_db', exist_ok=True)

    for i in range(len(df)):
        id = str(df['id'][i])
        pdf_path = os.path.join('pdf_db/', f'{id}.pdf')


        if os.path.exists(pdf_path):
            print(f'File {id}.pdf already exists.')
            continue

        url = f"https://arxiv.org/pdf/{id}.pdf"
        response = requests.get(url)
        
        if response.status_code == 200:
            with open(pdf_path, 'wb') as f:
                f.write(response.content)
            print(f'Downloaded {id}.pdf successfully.')
        else:
            print(f'Failed to download {id}.pdf.')


# df = pd.read_csv('../dataset/arxiv_metadata.csv', dtype={'id': str})

# check_and_download_pdf(df)
