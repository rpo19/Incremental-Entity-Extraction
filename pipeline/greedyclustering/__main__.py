import argparse
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from typing import List, Optional
import numpy as np
import base64
import pandas as pd
from sklearn_extra.cluster import KMedoids

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

app = FastAPI()

# greedy nearest neighbor clustering # doi 10.18653/v1/2021.acl-long.364
def cluster(scores, threshold):
    clusters = np.arange(scores.shape[0])
    for i, row in enumerate(scores):
        clusters[row > threshold] = clusters[i]
    return clusters

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

    # compute dot product (BLINK is trained on dot product) between embeddings
    current_encodings_np = np.array(current_encodings)
    scores = np.matmul(current_encodings_np, current_encodings_np.transpose())

    current_encodings = dict(zip(item.ids, current_encodings))

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

    for key, current_cluster in clusters.items():
        # title
        current_cluster['title'] = pd.Series(current_cluster['mentions']).value_counts().index[0]
        cluster_encodings = [current_encodings[i] for i in current_cluster['mentions_id']]
        current_cluster['center'] = vector_encode(KMedoids(n_clusters=1).fit(cluster_encodings).cluster_centers_)
        current_cluster['nelements'] = len(current_cluster['mentions'])

    return list(clusters.values())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host", type=str, default="127.0.0.1", help="host to listen at",
    )
    parser.add_argument(
        "--port", type=int, default="30305", help="port to listen at",
    )
    parser.add_argument(
        "--threshold", type=float, default=80.98388671875, help="Threshold for greedy NN clustering",
    )

    args = parser.parse_args()

    uvicorn.run(app, host = args.host, port = args.port)
