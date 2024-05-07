from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import string
import csv


# with open('arxiv_metadata.csv', 'r') as file:
#     csv_reader = csv.DictReader(file)
#     papers = [row for row in csv_reader]
# #print(papers)

# Sample dataset generation
sample_papers = [
    {
        'title': "A Survey of Deep Learning Techniques in Natural Language Processing",
        'abstract': "This paper provides a comprehensive survey of deep learning techniques used in natural language processing tasks such as sentiment analysis, machine translation, and named entity recognition."
    },
    {
        'title': "Applying Recurrent Neural Networks for Sentiment Analysis in NLP",
        'abstract': "This study explores the effectiveness of recurrent neural networks (RNNs) in sentiment analysis tasks within the domain of natural language processing. Experimental results demonstrate the superiority of RNN-based models over traditional methods."
    },
    {
        'title': "Challenges and Opportunities in Deep Learning for NLP",
        'abstract': "This paper discusses the challenges faced and the opportunities presented by deep learning approaches in natural language processing. It analyzes the current state-of-the-art techniques and identifies potential research directions."
    },
    {
        'title': "Syntax-aware Neural Machine Translation using Deep Learning",
        'abstract': "This research investigates syntax-aware neural machine translation models based on deep learning architectures. Experimental results show that incorporating syntactic information improves translation quality significantly."
    },
    {
        'title': "Deep Learning for Named Entity Recognition in Biomedical Texts",
        'abstract': "This work focuses on leveraging deep learning methods for named entity recognition tasks in biomedical texts. The proposed model achieves state-of-the-art performance on benchmark datasets."
    }
]




def preprocess(text):
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in string.punctuation]
    tokens = [word.lower() for word in tokens]
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    return tokens


def keyword_search(query, papers):
    query_tokens = set(preprocess(query))
    results = []
    for paper in papers:
        title_tokens = set(preprocess(paper['title']))
        abstract_tokens = set(preprocess(paper['abstract']))
        combined_tokens = title_tokens.union(abstract_tokens)
        matches = query_tokens.intersection(combined_tokens)
        if matches:
            results.append(paper)
    return results


query = "deeplearning in NLP"
matching_papers = keyword_search(query, sample_papers)

print("Matching papers:")
for paper in matching_papers:
    print("- Title:", paper['title'])
    print("  Abstract:", paper['abstract'])
    print()


