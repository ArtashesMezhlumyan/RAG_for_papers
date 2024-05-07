from hnswlib import Index
import numpy as np
from sentence_transformers import SentenceTransformer
import time
import pickle
import hnswlib
import torch
import warnings
from urllib3.exceptions import InsecureRequestWarning
from sentence_transformers import util
import pandas as pd


warnings.filterwarnings("ignore", category=InsecureRequestWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

def search_similar_abstracts(query, k=3):
    with open("../dataset/embeddings.pkl", "rb") as fIn:
        stored_data = pickle.load(fIn)
        stored_sentences = stored_data["sentences"]
        stored_embeddings = stored_data["embeddings"]
        stored_ids = stored_data["id"]  
    if isinstance(stored_embeddings, torch.Tensor):
        stored_embeddings = stored_embeddings.cpu().numpy() 

    dimension = stored_embeddings.shape[1]
    p = hnswlib.Index(space='cosine', dim=dimension)
    p.init_index(max_elements=10000, ef_construction=200, M=16)
    p.add_items(stored_embeddings)  
    p.set_ef(50) 

    query_embedding = model.encode([query])

    labels, distances = p.knn_query(query_embedding, k=k)
    similar_sentences_with_scores = [(stored_sentences[label], 1 - distance, str(stored_ids[label])) for label, distance in zip(labels[0], distances[0])]


    return similar_sentences_with_scores




def search_within_pdfs(query, selected_pdfs, k=10):
    with open("../pdf_manipulations/pdf_paragraphs_embeddings.pkl", "rb") as fIn:
        stored_data = pickle.load(fIn)
    returning_list = []
    for i in range(len(selected_pdfs)):
        next = selected_pdfs[i]
        selected_ids = [next]
        filtered_pdf_id_order = [uniq for id_, uniq in zip(stored_data["pdf_id"], stored_data["embedding"]) if id_ in selected_ids]
        filtered_pdf_ids = [id_ for id_ in stored_data["pdf_id"] if id_ in selected_ids]
        filtered_pdf_paragraph_embeddings = [embed for id_, embed in zip(stored_data["pdf_id"], stored_data["embedding"]) if id_ in selected_ids]
        filtered_pdf_paragraph_text = [text for id_, text in zip(stored_data["pdf_id"], stored_data["paragraph_text"]) if id_ in selected_ids]
        filtered_pdf_paragraph_embeddings = torch.tensor(filtered_pdf_paragraph_embeddings)

        
        if isinstance(filtered_pdf_paragraph_embeddings, torch.Tensor):
            filtered_pdf_paragraph_embeddings = filtered_pdf_paragraph_embeddings.cpu().numpy()  

        dimension = filtered_pdf_paragraph_embeddings.shape[1]
        p = hnswlib.Index(space='cosine', dim=dimension)
        p.init_index(max_elements=10000, ef_construction=200, M=16)
        p.add_items(filtered_pdf_paragraph_embeddings)  
        p.set_ef(50)  

        query_embedding = model.encode([query])

        labels, distances = p.knn_query(query_embedding, k=k)
        similar_sentences_with_scores = [(filtered_pdf_paragraph_text[label], 1 - distance, str(filtered_pdf_ids[label])) for label, distance in zip(labels[0], distances[0])]
        returning_list+=similar_sentences_with_scores

    return returning_list




# selected_pdfs = ['0704.0022', '0704.0021', '0704.0072']  # Example PDF IDs that were similar based on your initial abstract search
# query = "Tell me about nonlinear stochastic differential"

# similar_paragraphs_in_pdfs = search_within_pdfs(query, selected_pdfs)
# for snippet, score, arxiv_id in similar_paragraphs_in_pdfs:
#     print(f"Snippet: {snippet}")
#     print(f"Score: {score}")
#     print(f"ArXiv ID: {arxiv_id}")
#     print("-" * 40)  # Just to separate each entry for readability
#     print("---")


# start_time = time.time()
# new_sentence = "My sister's leg broke"
# search_similar_abstracts = search_similar_abstracts(new_sentence, k=3)

# print(search_similar_abstracts)
# # print("Top 3 similar sentences are:")
# # for i, sentence in enumerate(search_similar_abstracts):
# #     print(f"{i+1}. {sentence}")

# end_time = time.time()  # Records the end time
# elapsed_time = end_time - start_time  # Calculates the elapsed time

# print(f"The code ran for {elapsed_time} seconds.")

