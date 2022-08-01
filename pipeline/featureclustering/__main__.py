import argparse
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from typing import List, Optional
import numpy as np
import base64
import pandas as pd
from sklearn_extra.cluster import KMedoids
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pickle

def vector_encode(v):
    s = base64.b64encode(v).decode()
    return s

def vector_decode(s, dtype=np.float32):
    buffer = base64.b64decode(s)
    v = np.frombuffer(buffer, dtype=dtype)
    return v


class Item(BaseModel):
    ids: Optional[List[int]]
    mentions: List[str]
    embeddings: Optional[List[str]]
    encodings: Optional[List[str]]
    context_left: Optional[List[str]]
    context_right: Optional[List[str]]


app = FastAPI()

# greedy nearest neighbor clustering # doi 10.18653/v1/2021.acl-long.364
def cluster(scores, threshold):
    clusters = np.arange(scores.shape[0])
    for i, row in enumerate(scores):
        clusters[row > threshold] = clusters[i]
    return clusters

def vectorizer(mentions, contexts):
    bigram_vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 2), use_idf=False)
    bigram_vectorizer.fit(mentions)
    context_vectorizer = TfidfVectorizer(max_features=10000)
    context_vectorizer.fit(contexts)

    output_dict = {
        'bigram': bigram_vectorizer,
        'context': context_vectorizer,
    }
    return output_dict

def score(
    mentions,
    contexts,
    vectorizers,
    weights=(0.8, 0.2)
):

    print('Encoding mentions.')
    mention_vectors = vectorizers['bigram'].transform(mentions)

    print('Encoding contexts.')
    context_vectors = vectorizers['context'].transform(contexts)

    print('Scoring mentions.')
    mention_scores = linear_kernel(mention_vectors, mention_vectors)

    print('Scoring contexts.')
    context_scores = linear_kernel(context_vectors, context_vectors)

    scores = weights[0] * mention_scores + weights[1] * context_scores

    return scores

@app.post('/api/nilcluster')
async def cluster_mention(item: Item):
    global args

    total_clusters = []
    current_mentions = item.mentions
    if item.ids is not None:
        ids = item.ids
    else:
        ids = list(range(len(current_mentions)))
    if not item.embeddings:
        item.embeddings = item.encodings
    elif not item.encodings and not item.embeddings:
        raise Exception('Either "embeddings" or "encodings" field is required.')
    current_encodings = [vector_decode(e) for e in item.embeddings]

    mentions = item.mentions
    contexts = [left+' '+right for left, right in zip(item.context_left, item.context_right)]

    # load vectorizer from disk
    vectorizers = pickle.load('/home/app/models/vectorizer-incremental-dataset-dev100')

    current_encodings = np.array(current_encodings)
    scores = score(mentions, contexts, vectorizers)

    # clustering
    cluster_ids = cluster(scores, args.threshold)

    clusters = {}

    for i, cluster_id in enumerate(cluster_ids):
        if cluster_id not in clusters:
            clusters[cluster_id] = {}
            clusters[cluster_id]['mentions'] = []
            clusters[cluster_id]['mentions_id'] = []
        
        clusters[cluster_id]['mentions'].append(item.mentions[i])
        clusters[cluster_id]['mentions_id'].append(item.ids[i])

    for key, cluster in clusters.items():
        # title
        cluster['title'] = pd.Series(cluster['mentions']).value_counts().index[0]
        cluster_encodings = current_encodings[cluster['mention_ids']]
        cluster['center'] = KMedoids(n_clusters=1).fit(cluster_encodings).cluster_centers_
        cluster['nelements'] = len(cluster['mentions'])

    return clusters


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host", type=str, default="127.0.0.1", help="host to listen at",
    )
    parser.add_argument(
        "--port", type=int, default="30305", help="port to listen at",
    )
    parser.add_argument(
        "--threshold", type=float, default=100.0, help="Threshold for greedy NN clustering",
    )

    args = parser.parse_args()

    uvicorn.run(app, host = args.host, port = args.port)
