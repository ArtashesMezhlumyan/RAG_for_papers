from hnswlib import Index
import numpy as np
from sentence_transformers import SentenceTransformer
import time
import pickle
import hnswlib
import torch
import warnings
from urllib3.exceptions import InsecureRequestWarning

warnings.filterwarnings("ignore", category=InsecureRequestWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

def search_similar_sentences(query, k=3):
    with open("embeddings.pkl", "rb") as fIn:
        stored_data = pickle.load(fIn)
        stored_sentences = stored_data["sentences"]
        stored_embeddings = stored_data["embeddings"]

    # Ensure stored_embeddings is on CPU and converted to NumPy
    if isinstance(stored_embeddings, torch.Tensor):
        stored_embeddings = stored_embeddings.cpu().numpy()  # Convert to CPU and NumPy array

    dimension = stored_embeddings.shape[1]
    p = hnswlib.Index(space='cosine', dim=dimension)
    p.init_index(max_elements=10000, ef_construction=200, M=16)
    p.add_items(stored_embeddings)  # Now stored_embeddings is a NumPy array on CPU
    p.set_ef(50)  # Setting ef, which controls the recall

    query_embedding = model.encode([query])

    labels, distances = p.knn_query(query_embedding, k=k)
    similar_sentences_with_scores = [(stored_sentences[label], 1 - distance) for label, distance in zip(labels[0], distances[0])]


    return similar_sentences_with_scores





# new_sentence = "My sister's leg broke"
# top_similar_sentences = search_similar_sentences(new_sentence, k=3)

# print(top_similar_sentences)
# print("Top 3 similar sentences are:")
# for i, sentence in enumerate(top_similar_sentences):
#     print(f"{i+1}. {sentence}")