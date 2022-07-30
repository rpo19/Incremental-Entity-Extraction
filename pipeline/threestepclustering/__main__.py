import argparse
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from typing import List, Optional
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from fastDamerauLevenshtein import damerauLevenshtein
from scipy.spatial.distance import cdist
import numpy as np
import base64
from Packages.TimeEvolving import Cluster, compare_ecoding

def vector_encode(v):
    s = base64.b64encode(v).decode()
    return s

def vector_decode(s, dtype=np.float32):
    buffer = base64.b64decode(s)
    v = np.frombuffer(buffer, dtype=dtype)
    return v

def jacc_metric(x, y):
    x = set(x.lower().split())
    y = set(y.lower().split())

    intersection = len(x & y)
    union = len(x | y)

    jaccard_index = intersection / union

    jaccard_distance = 1 - jaccard_index

    return jaccard_distance

def jacc_lev_metric(x, y):
    x = set(x.lower().split())
    y = set(y.lower().split())

    intersection = 0

    for wx in x:
        for wy in y:
            normalized_lev_sim = 1 - dam_lev_metric(wx, wy) / max(len(wx), len(wy))
            intersection += normalized_lev_sim

    union = len(x) + len(y)

    jaccard_index = intersection / union

    jaccard_distance = 1 - jaccard_index

    return jaccard_distance


def dam_lev_metric(x, y):
    i, j = x[0], y[0]
    if len(i) < 4 or len(j) < 4:
        if i == j:
            return 0
        else:
            return damerauLevenshtein(i.lower(), j.lower(), similarity = False) + 3
    else:
        return damerauLevenshtein(i.lower(), j.lower(), similarity=False)

class Item(BaseModel):
    ids: Optional[List[int]]
    mentions: List[str]
    embeddings: Optional[List[str]]
    encodings: Optional[List[str]]

app = FastAPI()

@app.post('/api/nilcluster')
async def cluster_mention(item: Item):
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

    print('STEP 1')
    if len(current_mentions) == 1:
        cluster_numbers = np.zeros(1, dtype=np.int8)
    else:
        X = np.array(current_mentions).reshape(-1, 1)
        m_matrix = cdist(X, X, metric=dam_lev_metric)

        # clusterizator1 = DBSCAN(metric=dam_lev_metric, eps=1, min_samples=0, n_jobs=-1)
        clusterizator1 = AgglomerativeClustering(n_clusters=None, affinity='precomputed', #
                                                 distance_threshold=0.2,
                                                 linkage="single")

        cluster_numbers = clusterizator1.fit_predict(m_matrix)

    #Creo e vado a riempire un dizionario con chiave il numero del cluster e le menzioni all'interno del cluster, le entita corrispondenti
    #e l'encoding corrispondente ad: {0 : {entities: 'Milano', 'Milano', mentions: 'Milan', 'Milano', encodings:[[343][443]}}
    cee_dict = {k: {'mentions_id': [], 'mentions': [], 'encodings': [], 'sotto_clusters': None} for k in
                set(cluster_numbers)}

    for i, cluster in enumerate(cluster_numbers):
        cee_dict[cluster]['mentions_id'].append(ids[i])
        cee_dict[cluster]['mentions'].append(current_mentions[i])
        cee_dict[cluster]['encodings'].append(current_encodings[i])

    #STEP 2 - CLUSTERIZZAZIONE SEMANTICA - anche in questo caso agglomerativo: vado a raggruppare all'interno di ogni cluster,
    #creato nella fase precedente, gli elementi sulla base del loro encoding che tiene conto della semantica(BERT-encoding)

    print('STEP 2')
    #aggiungo l'appartenenza di questo clustering al campo 'sottocluster' il quale e' un array con i sotto_cluster di appartenenza
    #cee_list e' un array dove ogni elemento e' {entities: 'Milano', 'Milano', mentions: 'Milan', 'Milano', encodings:[[343][443]}
    cee_list = cee_dict.values()

    clusterizator2 = AgglomerativeClustering(n_clusters=None, affinity='cosine', distance_threshold=0.036,
                                             linkage="single")

    #ciclo sopra il numero del cluster 0 , 1 , 2 .. dipende quanti cluster ho trovato e salvato in cee_dict
    for cluster in cee_dict.keys():
        try:
            cee_dict[cluster]['sotto_clusters'] = clusterizator2.fit_predict(cee_dict[cluster]['encodings'])

        except ValueError:
            cee_dict[cluster]['sotto_clusters'] = np.zeros(1, dtype=np.int8)


    sottocluster_list = []
    #adesso cee_list e' sempre l'array in cui ogni elemento e' ogni cluster con entita'-menzioni- ma anche il campo sottocluster
    #ciclo sopra questo array: OGNI CELLA e' UN CLUSTER entities: 'Milano', 'Milano', mentions: 'Milan', 'Milano', encodings:[[343][443], sottocluster: [0,0]}
    for el in cee_list:
        #creo un dizionario ad ogni cluster con k il numero del sottocluster i esimo
        sotto_cluster = {k: Cluster() for k in set(el['sotto_clusters'])}

        for i, key in enumerate(el['sotto_clusters']):
            #in sottocluster i-esimo (key) quindi ad esempio sottocluster  0 aggiungo come valore l'entita' menzione e encoding
            #corrispondente ( un po' come il lavoro fatto prima ma ora per ogni sottocluster)
            sotto_cluster[key].add_element(mention=el['mentions'][i], entity='entity',
                                           encodings=el['encodings'][i], mentions_id=el['mentions_id'][i])
        #append alla liste sottocluster_list questo dizionario(1 dizionario per ogni cluster in cui all'interno abbiamo
        #il numero di sottocluster con le sue menzioni-entita'-encoding)
        sottocluster_list.append(sotto_cluster)

    sottocluster_list = [clusters_dict[key] for clusters_dict in sottocluster_list for key in clusters_dict]

    current_clusters = total_clusters + sottocluster_list
    #"FORSE" calcolo il centroide in ogni sottocluster
    sotto_encodings = [x.encodings_mean() for x in current_clusters]

    print('STEP 3')
    #STEP 3 - CLUSTERIZZAZIONE TRA I SOTTOCLUSTER SULLA BASE DEL CENTROIDE - UTILE PER I SINONIMI
    if len(sotto_encodings) == 1:
        cluster_numbers = np.zeros(1, dtype=np.int8)
    else:
        clusterizator3 = AgglomerativeClustering(n_clusters=None, affinity='cosine',
                                                 #distance_threshold=0.0155,
                                                 distance_threshold=0.05,
                                                 linkage="single")
        cluster_numbers = clusterizator3.fit_predict(sotto_encodings)
    final_clusters = {k: Cluster() for k in set(cluster_numbers)}
    last_key = list(set(final_clusters.keys()))[-1]
    for i, x in enumerate(current_clusters):
        if compare_ecoding(final_clusters[cluster_numbers[i]], x):
            final_clusters[cluster_numbers[i]] = final_clusters[cluster_numbers[i]] + x
        else:
            last_key = last_key + 1
            final_clusters[last_key] = x
    total_clusters = list(final_clusters.values())
    broken_cluster = []
    to_remove_cluster = []
    for cl_index, cl in enumerate(total_clusters):
        if len(set([men.lower() for men in cl.mentions])) > 25:
            X = np.array(cl.mentions).reshape(-1, 1)
            m_sub_matrix = cdist(X, X, metric=dam_lev_metric)
            br_clusterizator = AgglomerativeClustering(n_clusters=None, affinity='precomputed',
                                                       distance_threshold=0.2,
                                                       linkage="single")
            br_cluster_number = br_clusterizator.fit_predict(m_sub_matrix)
            br_cluster_dict = {k: Cluster() for k in set(br_cluster_number)}
            for i, cluster in enumerate(br_cluster_number):
                br_cluster_dict[cluster].add_element(cl.mentions[i], cl.entities[i], cl.encodings_list[i], cl.mentions_id[i])
            broken_cluster = broken_cluster + list(br_cluster_dict.values())
            to_remove_cluster.append(cl_index)
    for i in sorted(to_remove_cluster, reverse=True):
        del total_clusters[i]
    total_clusters = total_clusters + broken_cluster

    # getting centers
    for c in total_clusters:
        c.get_center()

    return total_clusters


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host", type=str, default="127.0.0.1", help="host to listen at",
    )
    parser.add_argument(
        "--port", type=int, default="30305", help="port to listen at",
    )

    args = parser.parse_args()

    uvicorn.run(app, host = args.host, port = args.port)
