from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd
import os
import numpy as np
import ast
import pickle
import torch
from pdf_to_text import extract_cleaned_text

model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=20,
    length_function = len
)

def vectorising_pdfs(df):
    paragraphs_info = []

    for i in range(len(df)):
        id = str(df['id'][i])
        print(f'Vectorising PDF {id}...')
        pdf_path = os.path.join('pdf_db/', f'{id}.pdf')
        
        try:
            text = extract_cleaned_text(pdf_path)
        except Exception as e:
            print(f'Error extracting text from {id}.pdf: {e}')
            text = f"Paper named {df['title']} cann't be downloaded from arxiv.org" 
        
        chunks = text_splitter.create_documents([text])
        chunk_texts = [str(chunk) for chunk in chunks]
        for j, chunk in enumerate(chunk_texts):
            encode_chunk = model.encode(chunk, convert_to_tensor=True)

            paragraphs_info.append({
                        "pdf_id_order": f"{i}_{j}",
                        "pdf_id": id,
                        "paragraph_text": chunk,
                        "embedding": encode_chunk.cpu().numpy()  
                    })
    
    df_paragraphs = pd.DataFrame(paragraphs_info)
    df_paragraphs.to_pickle("pdf_paragraphs_embeddings.pkl")

    return df_paragraphs


df = pd.read_csv('../dataset/arxiv_metadata.csv', dtype={'id': str})
vectorising_pdfs(df)

