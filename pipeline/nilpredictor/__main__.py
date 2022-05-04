import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from typing import List, Optional
import argparse
import textdistance
import statistics

# input: features
# output: NIL score

jaccardObj = textdistance.Jaccard(qval=None)
levenshteinObj = textdistance.Levenshtein(qval=None)

class Candidate(BaseModel):
    id: int
    indexer: int
    score: Optional[float]
    bi_score: Optional[float]

class Features(BaseModel):
    max_bi: Optional[float]
    max_cross: Optional[float]
    # text similarities
    title: Optional[str]
    mention: Optional[str]
    jaccard: Optional[float]
    levenshtein: Optional[float]
    # types
    mentionType: Optional[str]
    candidateType: Optional[str]
    candidateId: Optional[int]
    candidateIndexer: Optional[int]
    # stats
    topcandidates: Optional[List[Candidate]]

app = FastAPI()

@app.post('/api/nilprediction')
async def run(input: List[Features]):

    nil_X = pd.DataFrame()

    for i, features in enumerate(input):

        data = []
        index = []

        if features.max_bi:
            data.append(features.max_bi)
            index.append('max_bi')

        if features.max_cross:
            data.append(features.max_cross)
            index.append('max_cross')

        # process features
        _jacc, _leve = process_text_similarities(
            mention=features.mention, title=features.title, jaccard=features.jaccard, levenshtein=features.levenshtein)

        if _jacc is not None:
            data.append(_jacc)
            index.append('jaccard')

        if _leve is not None:
            data.append(_leve)
            index.append('levenshtein')

        # process types TODO

        # process stats TODO
        if features.topcandidates:
            # remove dummy candidates
            _topcandidates = [c for c in features.topcandidates if 'dummy' not in c]
            scores = [c.score for c in _topcandidates]

            stats = {
                'mean': statistics.mean(scores),
                'median': statistics.median(scores),
                'stdev': statistics.stdev(scores)
            }

            for i,v in stats.items():
                data.append(v)
                index.append(i)

        nil_X = nil_X.append(pd.Series(data=data, index=index, name=i))

    # run the model

    result = {}

    if nil_bi_model is not None:
        result['nil_score_bi'] = list(map(lambda x: x[1], nil_bi_model.predict_proba(nil_X[nil_bi_features])))

    if nil_model is not None:
        result['nil_score_cross'] = list(map(lambda x: x[1], nil_model.predict_proba(nil_X[nil_features])))

    return result

def process_text_similarities(mention=None, title=None, jaccard=None, levenshtein=None):
    if jaccard is None:
        if not (title is None and mention is None):
            mention_ = mention.lower()
            title_ = title.lower()
            jaccard = jaccardObj.normalized_similarity(mention_, title_)
    if levenshtein is None:
        if not (title is None and mention is None):
            mention_ = mention.lower()
            title_ = title.lower()
            levenshtein = levenshteinObj.normalized_similarity(mention_, title_)

    return jaccard, levenshtein


def load_nil_models(args, logger=None):
    if logger:
        logger.info('Loading nil bi model')
    if args.nil_bi_model is not None:
        with open(args.nil_bi_model, 'rb') as fd:
            nil_bi_model = pickle.load(fd)

        nil_bi_features = args.nil_bi_features.split(',')
    else:
        nil_bi_model = None
        nil_bi_features = None

    if logger:
        logger.info('Loading nil bi model')
    if args.nil_model is not None:
        with open(args.nil_model, 'rb') as fd:
            nil_model = pickle.load(fd)

        nil_features = args.nil_features.split(',')
    else:
        nil_model = None
        nil_features = None

    return nil_bi_model, nil_bi_features, nil_model, nil_features

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--host", type=str, default="127.0.0.1", help="host to listen at",
    )

    parser.add_argument(
        "--port", type=int, default="30303", help="port to listen at",
    )

    parser.add_argument(
        "--nil-bi-model", type=str, default=None, help="path to nil bi model",
    )

    parser.add_argument(
        "--nil-bi-features", type=str, default=None, help="features of the nil bi model (comma separated)",
    )

    parser.add_argument(
        "--nil-model", type=str, default=None, help="path to nil model",
    )

    parser.add_argument(
        "--nil-features", type=str, default=None, help="features of the nil model (comma separated)",
    )

    args = parser.parse_args()

    print('Loading nil models...')
    nil_bi_model, nil_bi_features, nil_model, nil_features = load_nil_models(args)
    print('Loading complete.')

    uvicorn.run(app, host = args.host, port = args.port)
