from openai import OpenAI
from hnsw_cosine import search_similar_sentences
import os
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)



os.environ['OPENAI_API_KEY'] = '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'
client = OpenAI()

def retrieved_texed(query):
    sentences_with_scores = search_similar_sentences(query)
    sentences_with_scores.sort(key=lambda x: x[1], reverse=True)

    # Extract top N main text
    top_n_main_text = [sentence[0] for sentence in sentences_with_scores]

    # Concatenate the top N main text into a single string
    main_text_string = '\n\n'.join(top_n_main_text)

    return main_text_string


def query_to_gpt(query):
    model="gpt-3.5-turbo-0125"
    message=[
        {"role": "system", "content": "You are a search engine experienced in academic papers"},
        {"role": "user", "content": retrieved_texed(query) + query}
    ]
    #
    response = client.chat.completions.create(
            model=model,
            messages=message,
            temperature=0.2,
            seed=92,
        )
    response_message = response.choices[0].message.content
    return response_message

print(query_to_gpt(query = "Tell me about evoulation of earth moon  system"))