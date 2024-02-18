# in theis code I am embedding the data and saving in to csv format

from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd
import os
import numpy as np
import ast
import pickle
import torch


model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

df = pd.read_csv('output.csv').loc[:2]


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=20,
    length_function = len
)


embedded_abstracts = []

for i in range(len(df['abstract'])):
    print(len(df['abstract'][i]))
    #deviding into chunks
    chunks = text_splitter.create_documents([df['abstract'][i]])
    
    #taking each chunk and embeding
    chunk_texts = [str(chunk) for chunk in chunks]

    encode_list = [model.encode(text,convert_to_tensor=True) for text in chunk_texts]
    #torch_vector = torch.cat(encode_list)
    summed_encoding = torch.zeros_like(encode_list[0])  # Initialize a tensor for summing
    for encoding in encode_list:
        summed_encoding += encoding
    normalised = summed_encoding/len(encode_list)
    embedded_abstracts.append(normalised)


appended_tensor = torch.stack(embedded_abstracts, dim=0)

with open("embeddings.pkl", "wb") as fOut:
        pickle.dump({"sentences": df['abstract'].tolist(), "embeddings": appended_tensor}, fOut, protocol=pickle.HIGHEST_PROTOCOL)
