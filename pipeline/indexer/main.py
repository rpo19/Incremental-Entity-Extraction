import argparse
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import numpy as np
import base64
from typing import List, Optional
from blink.indexer.faiss_indexer import DenseFlatIndexer, DenseHNSWFlatIndexer
import json
import psycopg
import os
# from annoy import AnnoyIndex

class _Index:
    def __init__(self, n):
        self.ntotal = n
# class AnnoyWrapper:
#     def __init__(self, annoyIndex):
#         self._index = annoyIndex
#         self.index = _Index(self._index.get_n_items())
#         self.index_type = 'annoy'
#     def search_knn(self, encodings, top_k):
#         candidates = []
#         scores = []
#         for v in encodings:
#             _c, _s = self._index.get_nns_by_vector(v, top_k, include_distances=True)
#             candidates.append(_c)
#             scores.append(_s)
#         return scores, candidates

def vector_encode(v):
    s = base64.b64encode(v).decode()
    return s

def vector_decode(s, dtype=np.float32):
    buffer = base64.b64decode(s)
    v = np.frombuffer(buffer, dtype=dtype)
    return v

class Input(BaseModel):
    encodings: List[str]
    top_k: int

indexes = []
rw_index = None

def id2url(wikipedia_id):
    global language
    if wikipedia_id > 0:
        return "https://{}.wikipedia.org/wiki?curid={}".format(language, wikipedia_id)
    else:
        return ""

app = FastAPI()

@app.post('/api/indexer/reset/rw')
async def reset():
    # reset rw index
    index_type = indexes[rw_index]['index_type']
    del indexes[rw_index]['indexer']
    if index_type == 'flat':
        indexes[rw_index]['indexer'] = DenseFlatIndexer(args.vector_size)
        indexes[rw_index]['indexer'].serialize(indexes[rw_index]['path'])
    else:
        raise Exception('Not implemented for index {}'.format(index_type))

    # reset db
    with dbconnection.cursor() as cur:
        print('deleting from db...')
        cur.execute("""
            DELETE
            FROM
                entities
            WHERE
                indexer = %s;
            """, (indexes[rw_index]['indexid'],))
    dbconnection.commit()

    return {'res': 'OK'}

@app.post('/api/indexer/search')
async def search(input_: Input):
    encodings = input_.encodings
    top_k = input_.top_k
    encodings = np.array([vector_decode(e) for e in encodings])
    all_candidates_4_sample_n = []
    for i in range(len(encodings)):
        all_candidates_4_sample_n.append([])
    for index in indexes:
        indexer = index['indexer']
        if indexer.index.ntotal == 0:
            scores = np.zeros((encodings.shape[0], top_k))
            candidates = -np.ones((encodings.shape[0], top_k)).astype(int)
        else:
            scores, candidates = indexer.search_knn(encodings, top_k)
        n = 0
        candidate_ids = set([id for cs in candidates for id in cs])
        with dbconnection.cursor() as cur:
            cur.execute("""
                SELECT
                    id, title, wikipedia_id, type_
                FROM
                    entities
                WHERE
                    id in ({}) AND
                    indexer = %s;
                """.format(','.join([str(int(id)) for id in candidate_ids])), (index['indexid'],))
            id2info = cur.fetchall()
        id2info = dict(zip(map(lambda x:x[0], id2info), map(lambda x:x[1:], id2info)))
        for _scores, _cands, _enc in zip(scores, candidates, encodings):

            # for each samples
            for _score, _cand in zip(_scores, _cands):
                raw_score = float(_score)
                _cand = int(_cand)
                if _cand == -1:
                    # -1 means no other candidates found
                    break
                # # compute dot product always (and normalized dot product)

                if _cand not in id2info:
                    # candidate removed from kb but not from index (requires to reconstruct the whole index)
                    all_candidates_4_sample_n[n].append({
                        'raw_score': -1000.0,
                        'id': _cand,
                        'wikipedia_id': 0,
                        'title': '',
                        'url': '',
                        'type_': '',
                        'indexer': index['indexid'],
                        'score': -1000.0,
                        'norm_score': -1000.0,
                        'dummy': 1
                    })
                    continue
                title, wikipedia_id, type_ = id2info[_cand]

                if index['index_type'] == 'flat':
                    embedding = indexer.index.reconstruct(_cand)
                elif index['index_type'] == 'hnsw':
                    embedding = indexer.index.reconstruct(_cand)[:-1]
                    _score = np.inner(_enc, embedding)
                elif index['index_type'] == 'annoy':
                    embedding = indexer._index.get_item_vector(_cand)
                else:
                    raise Exception('Should not happen.')

                # normalized dot product
                _enc_norm = np.linalg.norm(_enc)
                _embedding_norm = np.linalg.norm(embedding)
                _norm_factor = max(_enc_norm, _embedding_norm)**2
                _norm_score = _score / _norm_factor

                all_candidates_4_sample_n[n].append({
                        'raw_score': raw_score,
                        'id': _cand,
                        'wikipedia_id': wikipedia_id,
                        'title': title,
                        'url': id2url(wikipedia_id),
                        'type_': type_,
                        'indexer': index['indexid'],
                        'score': float(_score),
                        'norm_score': float(_norm_score)
                    })
            n += 1
    # sort
    for _sample in all_candidates_4_sample_n:
        _sample.sort(key=lambda x: x['score'], reverse=True)
    return all_candidates_4_sample_n

class Item(BaseModel):
    encoding: str
    wikipedia_id: Optional[int]
    title: str
    descr: Optional[str]
    type_: Optional[str]

@app.post('/api/indexer/add')
async def add(items: List[Item]):
    if rw_index is None:
        raise HTTPException(status_code=404, detail="No rw index!")

    # input: embeddings --> faiss
    # --> postgres
    # wikipedia_id ?
    # title
    # descr ?
    # embedding

    indexer = indexes[rw_index]['indexer']
    indexid = indexes[rw_index]['indexid']
    indexpath = indexes[rw_index]['path']

    # add to index
    embeddings = [vector_decode(e.encoding) for e in items]
    embeddings = np.stack(embeddings).astype('float32')
    indexer.index_data(embeddings)
    ids = list(range(indexer.index.ntotal - embeddings.shape[0], indexer.index.ntotal))
    # save index
    print(f'Saving index {indexid} to disk...')
    indexer.serialize(indexpath)

    # add to postgres
    with dbconnection.cursor() as cursor:
        with cursor.copy("COPY entities (id, indexer, wikipedia_id, title, descr, type_) FROM STDIN") as copy:
            for id, item in zip(ids, items):
                wikipedia_id = -1 if item.wikipedia_id is None else item.wikipedia_id
                copy.write_row((id, indexid, wikipedia_id, item.title, item.descr, item.type_))
    dbconnection.commit()

    return {
        'ids': ids,
        'indexer': indexid
    }

def load_models(args):
    assert args.index is not None, 'Error! Index is required.'
    for index in args.index.split(','):
        index_type, index_path, indexid, rorw = index.split(':')
        print('Loading {} index from {}, mode: {}...'.format(index_type, index_path, rorw))
        if os.path.isfile(index_path):
            if index_type == "flat":
                indexer = DenseFlatIndexer(1)
                indexer.deserialize_from(index_path)
            elif index_type == "hnsw":
                indexer = DenseHNSWFlatIndexer(1)
                indexer.deserialize_from(index_path)
            # elif index_type == 'annoy':
            #     _annoy_idx = AnnoyIndex(args.vector_size, 'dot')
            #     _annoy_idx.load(index_path)
            #     indexer = AnnoyWrapper(_annoy_idx)
            else:
                raise ValueError("Error! Unsupported indexer type! Choose from flat,hnsw.")
        else:
            if index_type == "flat":
                indexer = DenseFlatIndexer(args.vector_size)
            elif index_type == "hnsw":
                raise ValueError("Error! HNSW index File not Found! Cannot create a hnsw index from scratch.")
            else:
                raise ValueError("Error! Unsupported indexer type! Choose from flat,hnsw.")
        indexes.append({
            'indexer': indexer,
            'indexid': int(indexid),
            'path': index_path,
            'index_type': index_type
            })

        global rw_index
        if rorw == 'rw':
            assert rw_index is None, 'Error! Only one rw index is accepted.'
            rw_index = len(indexes) - 1 # last added

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # indexer
    parser.add_argument(
        "--index", type=str, default=None, help="comma separate list of paths to load indexes [type:path:indexid:ro/rw] (e.g: hnsw:index.pkl:0:ro,flat:index2.pkl:1:rw)",
    )
    parser.add_argument(
        "--host", type=str, default="127.0.0.1", help="host to listen at",
    )
    parser.add_argument(
        "--port", type=int, default="30301", help="port to listen at",
    )
    parser.add_argument(
        "--postgres", type=str, default=None, help="postgres url (e.g. postgres://user:password@localhost:5432/database)",
    )
    parser.add_argument(
        "--vector-size", type=int, default="1024", help="The size of the vectors", dest="vector_size",
    )
    parser.add_argument(
        "--language", type=str, default="en", help="Wikipedia language (en,it,...).",
    )

    args = parser.parse_args()

    assert args.postgres is not None, 'Error. postgres url is required.'
    dbconnection = psycopg.connect(args.postgres)

    language = args.language

    print('Loading indexes...')
    load_models(args)
    print('Loading complete.')

    uvicorn.run(app, host = args.host, port = args.port)
    dbconnection.close()
