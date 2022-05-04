# based on https://github.com/EntilZha/BLINK from EntilZha
import argparse
import sys
# blink is in ..
sys.path.append(".")

# +
import json

import numpy as np
import torch
import uvicorn
from colorama import init
from fastapi import FastAPI
from termcolor import colored
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from tqdm import tqdm

import blink.candidate_ranking.utils as utils
import blink.ner as NER
from blink.biencoder.biencoder import BiEncoderRanker, load_biencoder
from blink.biencoder.data_process import (get_candidate_representation,
                                          process_mention_data)
from blink.crossencoder.crossencoder import (CrossEncoderRanker,
                                             load_crossencoder)
from blink.crossencoder.data_process import prepare_crossencoder_data
from blink.crossencoder.train_cross import evaluate, modify
from blink.indexer.faiss_indexer import DenseFlatIndexer, DenseHNSWFlatIndexer

from pydantic import BaseModel
# -

from addict import Dict

from typing import List

# for nil models
import pickle
# for jaccard
import textdistance
#
import numpy as np
import pandas as pd

# patch for running fastapi in jupyter
import nest_asyncio


# +
class Item(BaseModel):
    text: str

class Sample(BaseModel):
    label:str
    label_id:int
    context_left: str
    context_right:str
    mention: str
    start_pos:int
    end_pos: int
    sent_idx:int
    #text: str

HIGHLIGHTS = [
    "on_red",
    "on_green",
    "on_yellow",
    "on_blue",
    "on_magenta",
    "on_cyan",
]


def _print_colorful_text(input_sentence, samples):
    init()  # colorful output
    msg = ""
    if samples and (len(samples) > 0):
        msg += input_sentence[0 : int(samples[0]["start_pos"])]
        for idx, sample in enumerate(samples):
            msg += colored(
                input_sentence[int(sample["start_pos"]) : int(sample["end_pos"])],
                "grey",
                HIGHLIGHTS[idx % len(HIGHLIGHTS)],
            )
            if idx < len(samples) - 1:
                msg += input_sentence[
                    int(sample["end_pos"]) : int(samples[idx + 1]["start_pos"])
                ]
            else:
                msg += input_sentence[int(sample["end_pos"]) :]
    else:
        msg = input_sentence
    print("\n" + str(msg) + "\n")


def _print_colorful_prediction(
    idx, sample, e_id, e_title, e_text, e_url, show_url=False
):
    print(colored(sample["mention"], "grey", HIGHLIGHTS[idx % len(HIGHLIGHTS)]))
    to_print = "id:{}\ntitle:{}\ntext:{}\n".format(e_id, e_title, e_text[:256])
    if show_url:
        to_print += "url:{}\n".format(e_url)
    print(to_print)


def _annotate(ner_model, input_sentences):
    ner_output_data = ner_model.predict(input_sentences)
    sentences = ner_output_data["sentences"]
    mentions = ner_output_data["mentions"]
    samples = []
    for mention in mentions:
        record = {}
        record["label"] = "unknown"
        record["label_id"] = -1
        # LOWERCASE EVERYTHING !
        record["context_left"] = sentences[mention["sent_idx"]][
            : mention["start_pos"]
        ].lower()
        record["context_right"] = sentences[mention["sent_idx"]][
            mention["end_pos"] :
        ].lower()
        record["mention"] = mention["text"].lower()
        record["start_pos"] = int(mention["start_pos"])
        record["end_pos"] = int(mention["end_pos"])
        record["sent_idx"] = mention["sent_idx"]
        samples.append(record)
    return samples


def _load_candidates(
    entity_catalogue, entity_encoding, faiss_index=None, index_path=None, logger=None
):
    # only load candidate encoding if not using faiss index
    if faiss_index is None:
        candidate_encoding = torch.load(entity_encoding)
        indexer = None
    else:
        if logger:
            logger.info("Using faiss index to retrieve entities.")
        candidate_encoding = None
        assert index_path is not None, "Error! Empty indexer path."
        if faiss_index == "flat":
            indexer = DenseFlatIndexer(1)
        elif faiss_index == "hnsw":
            indexer = DenseHNSWFlatIndexer(1)
        else:
            raise ValueError("Error! Unsupported indexer type! Choose from flat,hnsw.")
        indexer.deserialize_from(index_path)

    # load all the 5903527 entities
    title2id = {}
    id2title = {}
    id2text = {}
    wikipedia_id2local_id = {}
    local_idx = 0
    with open(entity_catalogue, "r") as fin:
        lines = fin.readlines()
        for line in lines:
            entity = json.loads(line)

            if "idx" in entity:
                split = entity["idx"].split("curid=")
                if len(split) > 1:
                    wikipedia_id = int(split[-1].strip())
                else:
                    wikipedia_id = entity["idx"].strip()

                assert wikipedia_id not in wikipedia_id2local_id
                wikipedia_id2local_id[wikipedia_id] = local_idx

            title2id[entity["title"]] = local_idx
            id2title[local_idx] = entity["title"]
            id2text[local_idx] = entity["text"]
            local_idx += 1
    return (
        candidate_encoding,
        title2id,
        id2title,
        id2text,
        wikipedia_id2local_id,
        indexer,
    )


def __map_test_entities(test_entities_path, title2id, logger):
    # load the 732859 tac_kbp_ref_know_base entities
    kb2id = {}
    missing_pages = 0
    n = 0
    with open(test_entities_path, "r") as fin:
        lines = fin.readlines()
        for line in lines:
            entity = json.loads(line)
            if entity["title"] not in title2id:
                missing_pages += 1
            else:
                kb2id[entity["entity_id"]] = title2id[entity["title"]]
            n += 1
    if logger:
        logger.info("missing {}/{} pages".format(missing_pages, n))
    return kb2id


def __load_test(test_filename, kb2id, wikipedia_id2local_id, logger):
    test_samples = []
    with open(test_filename, "r") as fin:
        lines = fin.readlines()
        for line in lines:
            record = json.loads(line)
            record["label"] = str(record["label_id"])

            # for tac kbp we should use a separate knowledge source to get the entity id (label_id)
            if kb2id and len(kb2id) > 0:
                if record["label"] in kb2id:
                    record["label_id"] = kb2id[record["label"]]
                else:
                    continue

            # check that each entity id (label_id) is in the entity collection
            elif wikipedia_id2local_id and len(wikipedia_id2local_id) > 0:
                try:
                    key = int(record["label"].strip())
                    if key in wikipedia_id2local_id:
                        record["label_id"] = wikipedia_id2local_id[key]
                    else:
                        continue
                except:
                    continue

            # LOWERCASE EVERYTHING !
            record["context_left"] = record["context_left"].lower()
            record["context_right"] = record["context_right"].lower()
            record["mention"] = record["mention"].lower()
            test_samples.append(record)

    if logger:
        logger.info("{}/{} samples considered".format(len(test_samples), len(lines)))
    return test_samples


def _get_test_samples(
    test_filename, test_entities_path, title2id, wikipedia_id2local_id, logger
):
    kb2id = None
    if test_entities_path:
        kb2id = __map_test_entities(test_entities_path, title2id, logger)
    test_samples = __load_test(test_filename, kb2id, wikipedia_id2local_id, logger)
    return test_samples


def _process_biencoder_dataloader(samples, tokenizer, biencoder_params):
    _, tensor_data = process_mention_data(
        samples,
        tokenizer,
        biencoder_params["max_context_length"],
        biencoder_params["max_cand_length"],
        silent=True,
        logger=None,
        debug=biencoder_params["debug"],
    )
    sampler = SequentialSampler(tensor_data)
    dataloader = DataLoader(
        tensor_data, sampler=sampler, batch_size=biencoder_params["eval_batch_size"]
    )
    return dataloader


def _run_biencoder(biencoder, dataloader, candidate_encoding, top_k=100, indexer=None):
    biencoder.model.eval()
    labels = []
    nns = []
    all_scores = []
    all_encodings = []
    for batch in tqdm(dataloader):
        context_input, _, label_ids = batch
        with torch.no_grad():
            if indexer is not None:
                context_encoding = biencoder.encode_context(context_input).numpy()
                context_encoding = np.ascontiguousarray(context_encoding)
                scores, indicies = indexer.search_knn(context_encoding, top_k)
                if isinstance(indexer, DenseHNSWFlatIndexer):
                    # compute dot products
                    # use indicies to find ids of candidates
                    # then get their encodings with candidate_encoding
                    # then calculate dot-prod with context_enc
                    # reorder ?
                    pass
            else:
                raise Exception('Indexer should be used.')
                scores = biencoder.score_candidate(
                    context_input, None, cand_encs=candidate_encoding  # .to(device)
                )
                scores, indicies = scores.topk(top_k)
                scores = scores.data.numpy()
                indicies = indicies.data.numpy()

        labels.extend(label_ids.data.numpy())
        nns.extend(indicies)
        all_scores.extend(scores)
        all_encodings.extend([e.tolist() for e in context_encoding])
    return labels, nns, all_scores, all_encodings


def _process_crossencoder_dataloader(context_input, label_input, crossencoder_params):
    tensor_data = TensorDataset(context_input, label_input)
    sampler = SequentialSampler(tensor_data)
    dataloader = DataLoader(
        tensor_data, sampler=sampler, batch_size=crossencoder_params["eval_batch_size"]
    )
    return dataloader


def _run_crossencoder(crossencoder, dataloader, logger, context_len, device="cuda"):
    crossencoder.model.eval()
    accuracy = 0.0
    crossencoder.to(device)

    res = evaluate(crossencoder, dataloader, device, logger, context_len, silent=False)
    accuracy = res["normalized_accuracy"]
    logits = res["logits"]

    predictions = np.argsort(logits, axis=1)
    return accuracy, predictions, logits


def load_models(args, logger=None):
    # load biencoder model
    if logger:
        logger.info("loading biencoder model")
    with open(args.biencoder_config) as json_file:
        biencoder_params = json.load(json_file)
        biencoder_params["path_to_model"] = args.biencoder_model
    biencoder = load_biencoder(biencoder_params)

    crossencoder = None
    crossencoder_params = None
    if not args.fast:
        # load crossencoder model
        if logger:
            logger.info("loading crossencoder model")
        with open(args.crossencoder_config) as json_file:
            crossencoder_params = json.load(json_file)
            crossencoder_params["path_to_model"] = args.crossencoder_model
        crossencoder = load_crossencoder(crossencoder_params)

    # load candidate entities
    if logger:
        logger.info("loading candidate entities")
    (
        candidate_encoding,
        title2id,
        id2title,
        id2text,
        wikipedia_id2local_id,
        faiss_indexer,
    ) = _load_candidates(
        args.entity_catalogue,
        args.entity_encoding,
        faiss_index=args.faiss_index,
        index_path=args.index_path,
        logger=logger,
    )

    id2url = {
        v: "https://en.wikipedia.org/wiki?curid=%s" % k
        for k, v in wikipedia_id2local_id.items()
    }

    return (
        biencoder,
        biencoder_params,
        crossencoder,
        crossencoder_params,
        candidate_encoding,
        title2id,
        id2title,
        id2text,
        wikipedia_id2local_id,
        faiss_indexer,
        id2url
    )

def link_text(text):
    # Identify mentions
    samples = _annotate(ner_model, [text])
    _print_colorful_text(text, samples)
    if len(samples) == 0:
        return []
    else:
        return link_samples(samples, models, nil_bi_models, nil_models, logger)

def link_samples(samples, models, nil_bi_models=None, nil_models=None, logger=None):
    biencoder = models[0]
    biencoder_params = models[1]

    candidate_encoding = models[4]
    faiss_indexer = models[9]

    id2title = models[6]
    id2text = models[7]
    id2url = models[10]

    # don't look at labels
    keep_all = True

    if logger:
        # prepare the data for biencoder
        logger.info("preparing data for biencoder")
    dataloader = _process_biencoder_dataloader(
        samples, biencoder.tokenizer, biencoder_params
    )

    if logger:
        # run biencoder
        logger.info("run biencoder")
    top_k = args.top_k
    labels, nns, scores, encodings = _run_biencoder(
        biencoder,
        dataloader,
        candidate_encoding,
        top_k,
        faiss_indexer,
    )

    # nil prediction
    if nil_bi_models:
        nil_bi_model = nil_bi_models[0]
        nil_bi_features = nil_bi_models[1]

        nil_X = pd.DataFrame()
        nil_X['max'] = list(map(lambda x: x[0], scores))

        def _get_jaccard(samples, nns):
            # todo replace `-` with ` ` for jaccard ?
            jacc = textdistance.Jaccard(qval=None)

            return list(map(lambda x: jacc.normalized_similarity(x[0]['mention'].lower(), id2title[x[1][0]].lower()),
                   zip(samples, nns)))

        nil_X['jaccard'] = _get_jaccard(samples, nns)

        nil_bi_p = list(map(lambda x: x[1], nil_bi_model.predict_proba(nil_X[nil_bi_features])))
    else:
        nil_bi_p = [1] * len(nns)



    # print biencoder prediction
    idx = 0
    linked_entities = []
    for entity_list, sample, _nil_p, _scores, _encoding in zip(nns, samples, nil_bi_p, scores, encodings):

        candidates = []
        for e_id, _score in zip(entity_list, _scores):
            e_title = id2title[e_id]
            e_text = id2text[e_id]
            e_url = id2url[e_id]
            candidates.append({
                "entity": {
                    "e_id": int(e_id),
                    "e_title": e_title,
                    "e_url": e_url,
                    "e_text": e_text
                },
                "score": float(_score)
            })


        e_id = entity_list[0]
        e_title = id2title[e_id]
        e_text = id2text[e_id]
        e_url = id2url[e_id]

        linked_entities.append(
            {
                "idx": idx,
                "sample": sample,
                "entity_id": e_id.item(),
                "entity_title": e_title,
                "entity_text": e_text,
                "url": e_url,
                "fast": True,
                "_nil_p": _nil_p,
                "candidates": candidates,
                "encoding": _encoding
            }
        )
        idx += 1

    if args.fast:
        # use only biencoder
        return {"samples": samples, "linked_entities": linked_entities}
    else:
        crossencoder = models[2]
        crossencoder_params = models[3]

        # prepare crossencoder data
        context_input, candidate_input, label_input = prepare_crossencoder_data(
            crossencoder.tokenizer, samples, labels, nns, id2title, id2text, keep_all,
        )

        context_input = modify(
            context_input, candidate_input, crossencoder_params["max_seq_length"]
        )

        dataloader = _process_crossencoder_dataloader(
            context_input, label_input, crossencoder_params
        )

        # run crossencoder and get accuracy
        accuracy, index_array, unsorted_scores = _run_crossencoder(
            crossencoder,
            dataloader,
            logger,
            context_len=biencoder_params["max_context_length"],
        )

        # nil prediction
        if nil_models:
            nil_model = nil_models[0]
            nil_features = nil_models[1]

            nil_X = pd.DataFrame()

            def _helper_f(index_array, unsorted_scores):
                for _index, _scores in zip(index_array, unsorted_scores):
                    _top_ranked_entity_idx = _index[-1]
                    _max_score = _scores[_top_ranked_entity_idx]

                    yield _max_score

            max_scores = list(_helper_f(index_array, unsorted_scores))

            nil_X['max'] = max_scores

            def _get_jaccard(samples, nns):
                # todo replace `-` with ` ` for jaccard ?
                jacc = textdistance.Jaccard(qval=None)

                return list(map(lambda x: jacc.normalized_similarity(x[0]['mention'].lower(), id2title[x[1][0]].lower()),
                    zip(samples, nns)))

            nil_X['jaccard'] = _get_jaccard(samples, nns)

            nil_p = list(map(lambda x: x[1], nil_model.predict_proba(nil_X[nil_features])))
        else:
            nil_p = [1] * len(nns)

        scores = []
        predictions = []
        linked_entities = []
        for entity_list, index_list, scores_list, sample, _nil_p, _encoding in zip(
            nns, index_array, unsorted_scores, samples, nil_p, encodings
        ):

            best_e_id = entity_list[index_list[-1]]

            index_list = index_list.tolist()

            # descending order
            index_list.reverse()

            candidates = []
            for index in index_list:
                e_id = int(entity_list[index])
                e_title = id2title[e_id]
                e_text = id2text[e_id]
                e_url = id2url[e_id]
                candidates.append({
                    "entity": {
                        "e_id": int(e_id),
                        "e_title": e_title,
                        "e_url": e_url,
                        "e_text": e_text
                    },
                    "score": float(scores_list[index])
                })

            best_e_title = id2title[best_e_id]
            best_e_text = id2text[best_e_id]
            best_e_url = id2url[best_e_id]
            linked_entities.append(
                {
                    "idx": idx,
                    "sample": sample,
                    "entity_id": int(best_e_id),
                    "entity_title": best_e_title,
                    "entity_text": best_e_text,
                    "score": float(scores_list[0]),
                    "url": best_e_url,
                    "fast": False,
                    "_nil_p": _nil_p,
                    "candidates": candidates,
                    "encoding": _encoding
                }
            )
            idx += 1


        return {"samples": samples, "linked_entities": linked_entities}



def create_app():
    app = FastAPI()

    @app.post("/api/entity-link/text")
    async def entity_link_text(item: Item):
        return link_text(item.text)

    @app.post("/api/entity-link/samples")
    async def entity_link_samples(samples: List[Sample]):
        samples = [dict(s) for s in samples]
        res = link_samples(samples, models, nil_bi_models, nil_models, logger)
        return res

    return app


# -

def load_nil_models(args, logger=None):
    if logger:
        logger.info('Loading nil bi model')
    with open(args.nil_bi_model, 'rb') as fd:
        nil_bi_model = pickle.load(fd)

    nil_bi_features = args.nil_bi_features.split(',')

    if logger:
        logger.info('Loading nil bi model')
    with open(args.nil_model, 'rb') as fd:
        nil_model = pickle.load(fd)

    nil_features = args.nil_features.split(',')

    return ((nil_bi_model, nil_bi_features), (nil_model, nil_features))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--interactive", "-i", action="store_true", help="Interactive mode."
    )

    # test_data
    parser.add_argument(
        "--test_mentions", dest="test_mentions", type=str, help="Test Dataset."
    )
    parser.add_argument(
        "--test_entities", dest="test_entities", type=str, help="Test Entities."
    )

    # biencoder
    parser.add_argument(
        "--biencoder_model",
        dest="biencoder_model",
        type=str,
        default="models/biencoder_wiki_large.bin",
        help="Path to the biencoder model.",
    )
    parser.add_argument(
        "--biencoder_config",
        dest="biencoder_config",
        type=str,
        default="models/biencoder_wiki_large.json",
        help="Path to the biencoder configuration.",
    )
    parser.add_argument(
        "--entity_catalogue",
        dest="entity_catalogue",
        type=str,
        # default="models/tac_entity.jsonl",  # TAC-KBP
        default="models/entity.jsonl",  # ALL WIKIPEDIA!
        help="Path to the entity catalogue.",
    )
    parser.add_argument(
        "--entity_encoding",
        dest="entity_encoding",
        type=str,
        # default="models/tac_candidate_encode_large.t7",  # TAC-KBP
        default="models/all_entities_large.t7",  # ALL WIKIPEDIA!
        help="Path to the entity catalogue.",
    )

    # crossencoder
    parser.add_argument(
        "--crossencoder_model",
        dest="crossencoder_model",
        type=str,
        default="models/crossencoder_wiki_large.bin",
        help="Path to the crossencoder model.",
    )
    parser.add_argument(
        "--crossencoder_config",
        dest="crossencoder_config",
        type=str,
        default="models/crossencoder_wiki_large.json",
        help="Path to the crossencoder configuration.",
    )

    parser.add_argument(
        "--top_k",
        dest="top_k",
        type=int,
        default=10,
        help="Number of candidates retrieved by biencoder.",
    )

    # output folder
    parser.add_argument(
        "--output_path",
        dest="output_path",
        type=str,
        default="output",
        help="Path to the output.",
    )

    parser.add_argument(
        "--fast", dest="fast", action="store_true", help="only biencoder mode"
    )

    parser.add_argument(
        "--show_url",
        dest="show_url",
        action="store_true",
        help="whether to show entity url in interactive mode",
    )

    parser.add_argument(
        "--faiss_index", type=str, default=None, help="whether to use faiss index",
    )

    parser.add_argument(
        "--index_path", type=str, default=None, help="path to load indexer",
    )

    parser.add_argument(
        "--host", type=str, default="127.0.0.1", help="host to listen at",
    )

    parser.add_argument(
        "--port", type=int, default="30300", help="port to listen at",
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

    logger = utils.get_logger(args.output_path)

    models = load_models(args, logger)

    local_id2wikipedia_id = None

    wikipedia_id2local_id = models[8]
    if hasattr(args, 'save_encodings') and args.save_encodings:
        local_id2wikipedia_id = {}
        for k,v in wikipedia_id2local_id.items():
            local_id2wikipedia_id[v] = k

    ner_model = NER.get_model()

    models = load_models(args, logger)

    nil_bi_models, nil_models = load_nil_models(args, logger)
    nil_bi_models = None # disable

    # patch for running fastapi in jupyter
    #nest_asyncio.apply()

    app = create_app()
    uvicorn.run(app, host=args.host, port=args.port)
