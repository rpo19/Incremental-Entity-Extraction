from email.policy import default
import click
from tqdm import tqdm
import requests
import numpy as np
import pandas as pd
import json
import sys
import os
import base64
from pprint import pprint
import bcubed
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import statistics
import textdistance
from sklearn_extra.cluster import KMedoids

### TODO move all them behind a single proxy and set configurable addresses
biencoder = 'http://localhost:30300/api/blink/biencoder' # mention # entity
biencoder_mention = f'{biencoder}/mention'
biencoder_entity = f'{biencoder}/entity'
crossencoder = 'http://localhost:30302/api/blink/crossencoder'
indexer = 'http://localhost:30301/api/indexer' # search # add
indexer_search = f'{indexer}/search'
indexer_add = f'{indexer}/add'
indexer_reset = f'{indexer}/reset/rw'
nilpredictor = 'http://localhost:30303/api/nilprediction'
nilcluster = 'http://localhost:30305/api/nilcluster'
###

###
context_right = 'context_right'
context_left = 'context_left'
mention = 'mention'
###

# wikipedia_id = mode(cluster.wikipedia_ids)
added_entities = pd.DataFrame()
# clusters with multiple modes
prev_clusters = pd.DataFrame()

def vector_encode(v):
    s = base64.b64encode(v).decode()
    return s

def vector_decode(s, dtype=np.float32):
    buffer = base64.b64decode(s)
    v = np.frombuffer(buffer, dtype=dtype)
    return v

def _bc_get_stats(x, remove_correct=False, scores_col='scores', nns_col='nns', labels_col='labels', top_k=100):
    scores = x[scores_col]
    nns = x[nns_col]
    if isinstance(scores, str):
        scores = np.array(json.loads(scores))
    if isinstance(nns, str):
        nns = np.array(json.loads(nns))

    assert len(scores) == len(nns)
    scores = scores.copy()

    sort_scores_i = np.argsort(scores)[::-1]
    scores = np.array(scores)
    scores = scores[sort_scores_i][:top_k]

    nns = nns.copy()
    nns = np.array(nns)
    nns = nns[sort_scores_i][:top_k]

    correct = None
    if x[labels_col] in nns:
        # found correct entity
        i_correct = list(nns).index(x[labels_col])
        correct = scores[i_correct]

    second = sorted(scores, reverse=True)[1]

    _stats = {
        "correct": correct,
        "max": max(scores),
        "second": second,
        "secondiff": max(scores) - second,
        "min": min(scores),
        "mean": statistics.mean(scores),
        "median": statistics.median(scores),
        "stdev": statistics.stdev(scores)
    }
    return _stats


def _bi_get_stats(x, remove_correct=False, top_k=100):
    return _bc_get_stats(x, remove_correct=remove_correct, scores_col='scores', top_k=top_k)

def prepare_for_nil_prediction_train(df):
    df['top_id'] = df['candidates'].apply(lambda x: x[0]['wikipedia_id'])
    df['top_title'] = df['candidates'].apply(lambda x: x[0]['title'])
    df[['scores', 'nns']] = df.apply(lambda x: {'scores': [i['score'] for i in x['candidates'] if i['wikipedia_id'] > 0], 'nns': [i['wikipedia_id'] for i in x['candidates'] if i['wikipedia_id'] > 0]}, result_type='expand', axis=1)
    #df['nns'] = df['candidates'].apply(lambda x: [i['wikipedia_id'] for i in x])
    df['labels'] = df.eval('~NIL and Wikipedia_ID == top_id').astype(int)

    stats = df.apply(_bi_get_stats, axis=1, result_type='expand')
    df[stats.columns] = stats

    levenshtein = textdistance.Levenshtein(qval=None)
    jaccard = textdistance.Jaccard(qval=None)
    df['levenshtein'] = df.apply(lambda x: levenshtein.normalized_similarity(x['mention'].lower(), x['top_title'].lower()), axis=1)
    df['jaccard'] = df.apply(lambda x: jaccard.normalized_similarity(x['mention'].lower(), x['top_title'].lower()), axis=1)

    return df

def prepare_for_nil_prediction(x):
    c = x['candidates']

    is_nil = False
    features = {}

    if len(c) == 0:
        is_nil = True
        return is_nil, features

    is_cross = 'is_cross' in c[0] and c[0]['is_cross']

    features = {}
    if not is_cross:
        # bi only
        features['max_bi'] = c[0]['score']
    else:
        # cross
        if 'bi_score' in c[0]:
            features['max_bi'] = c[0]['bi_score']
        features['max_cross'] = c[0]['score']

    features['mention'] = x[mention]
    features['title'] = c[0]['title']
    features['topcandidates'] = c

    return is_nil, features

def run_batch(batch, data, no_add, save_path, prepare_for_nil_prediction_train_flag, correct_steps,
        biencoder=biencoder,
        biencoder_mention=biencoder_mention,
        biencoder_entity=biencoder_entity,
        crossencoder=crossencoder,
        indexer=indexer,
        indexer_search=indexer_search,
        indexer_add=indexer_add,
        nilpredictor=nilpredictor,
        nilcluster=nilcluster):

    global added_entities
    global prev_clusters

    print('Run batch', batch)

    # Evaluation
    # TODO use also top_title?
    report = {}
    report['batch'] = batch
    report['size'] = data.shape[0]

    # ## Entity Linking

    # ### Encoding
    print('Encoding entities...')
    res_biencoder = requests.post(biencoder_mention,
            json=data[[
                'mention',
                'context_left',
                'context_right'
                ]].to_dict(orient='records'))

    if res_biencoder.ok:
        data['encoding'] = res_biencoder.json()['encodings']
    else:
        print('Biencoder ERROR')
        print(res_biencoder)
        raise Exception('Biencoder ERROR')

    print('Encoded {} entities.'.format(data.shape[0]))

    # ### Retrieval
    print('retrieval')
    body = {
        'encodings': data['encoding'].values.tolist(),
        'top_k': 10
    }
    res_indexer = requests.post(indexer_search, json=body)

    if res_indexer.ok:
        candidates = res_indexer.json()
    else:
        print('ERROR with the indexer.')
        print(res_indexer)
        print(res_indexer.json())

    if len(candidates) == 0 or len(candidates[0]) == 0:
        print('No candidates received.')

    data['candidates'] = candidates

    ######### < evaluate linking

    data['top_Wikipedia_ID'] = data.apply(lambda x: x['candidates'][0]['wikipedia_id'], axis=1)

    ## Linking
    def eval_linking_helper(x):
        candidate_ids = [i['wikipedia_id'] for i in x['candidates']]
        try:
            return candidate_ids.index(x['Wikipedia_ID']) + 1 # starting from recall@1
        except ValueError:
            return -1

    not_nil = data.query('~NIL').copy()

    not_nil['linking_found_at'] = not_nil.apply(eval_linking_helper, axis=1)

    for i in [1,2,3,5,10,30,100]:
        report[f'linking_recall@{i}'] = not_nil.eval(f'linking_found_at > 0 and linking_found_at <= {i}').sum() / not_nil.shape[0]

    ######### eval linking >

    if prepare_for_nil_prediction_train_flag:
        data = prepare_for_nil_prediction_train(data)

        os.makedirs(save_path, exist_ok=True)
        batch_basename = os.path.splitext(os.path.basename(batch))[0]
        outdata = os.path.join(save_path, '{}_outdata.pickle'.format(batch_basename))
        data.to_pickle(outdata)

        return {}

    data[['is_nil', 'nil_features']] = data.apply(prepare_for_nil_prediction, axis=1, result_type='expand')

    # ## NIL prediction
    print('nil prediction')
    # prepare fields (default NIL)
    data['nil_score'] = np.zeros(data.shape[0])

    if 'dropped' not in report:
        report['dropped'] = []
    dropped = 0

    if correct_steps:
        # correct linking: modify candidates so that the correct candidate is the first.
        def correct_linking_candidates(x):
            global dropped
            # put correct candidate first
            # remove candidates whose score is higher than the correct
            # repeat the worst candidates to reach the same size of candidates
            if x['linking_found_at'] > 0:
                idx = x['linking_found_at'] - 1
                prev_len = len(x['candidates'])
                x['candidates'] = x['candidates'][idx:]
                if len(x['candidates']) == 1:
                    # correct candidate at the end: drop
                    dropped += 1
                    return None
                worst = x['candidates'][-1]
                for i in range(prev_len - len(x['candidates'])):
                    x['candidates'].append(worst)
                assert len(x['candidates']) == prev_len
            else:
                # if not found and not nil: drop
                dropped += 1
                return None

        data['linking_found_at'] = data.apply(eval_linking_helper, axis=1)
        data['candidates'] = data.apply(correct_linking_candidates)
        data = data.dropna(subset='candidates')

        report['dropped'].append(dropped)

    not_yet_nil = data.query('is_nil == False')

    if not_yet_nil.shape[0] > 0:
        res_nilpredictor = requests.post(nilpredictor, json=not_yet_nil['nil_features'].values.tolist())
        if res_nilpredictor.ok:
            # TODO use cross if available
            nil_scores_bi = np.array(res_nilpredictor.json()['nil_score_bi'])
            nil_scores_cross = np.array(res_nilpredictor.json()['nil_score_bi'])
        else:
            print('ERROR during NIL prediction')
            print(res_nilpredictor)
            print(res_nilpredictor.json())

    data.loc[not_yet_nil.index, 'nil_score'] = nil_scores_bi
    data.loc[not_yet_nil.index, 'nil_score_cross'] = nil_scores_cross

    nil_threshold = 0.5
    # if below threshold --> is NIL
    data['is_nil'] = data['nil_score'].apply(lambda x: x < nil_threshold)

    print('Estimated {} entities as NOT NIL'.format(data.eval('is_nil == False').sum()))
    print('Estimated {} entities as NIL'.format(data.eval('is_nil == True').sum()))

    data['top_title'] = data['candidates'].apply(lambda x: x[0]['title'])

    ###### < eval nil pred
    ## NIL prediction
    should_be_nil_bool = data['NIL'] & ~data['Wikipedia_ID'].isin(prev_added_entities.index)
    data['should_be_nil'] = should_be_nil_bool

    report['nil_prediction'] = classification_report(should_be_nil_bool, data['is_nil'], output_dict=True)

    ### NIL prediction mitigated
    ### consider correct also when linking is not correct but is_nil
    data['NIL_mitigated'] = data.eval('should_be_nil or top_Wikipedia_ID != Wikipedia_ID')
    report['nil_prediction_mitigated'] = classification_report(data['NIL_mitigated'], data['is_nil'], output_dict=True)
    tn, fp, fn, tp = confusion_matrix(should_be_nil_bool, data['is_nil']).ravel()
    report['nil_prediction_cm'] = {
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tp': tp
    }

    tn, fp, fn, tp = confusion_matrix(should_be_nil_bool, data['is_nil'], normalize = "true").ravel()
    report['nil_prediction_cm_normalized'] = {
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tp': tp
    }
    tn, fp, fn, tp = confusion_matrix(data['NIL_mitigated'], data['is_nil']).ravel()
    report['nil_prediction_mitigated_cm'] = {
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tp': tp
    }

    tn, fp, fn, tp = confusion_matrix(data['NIL_mitigated'], data['is_nil'], normalize = "true").ravel()
    report['nil_prediction_mitigated_cm_normalized'] = {
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tp': tp
    }
    ###### eval nil pred >

    if correct_steps:
        data['is_nil'] = data['should_be_nil']

    # necessary for evaluation
    prev_added_entities = added_entities.copy()

    # TODO consider correcting added_entities?
    added_entities = pd.concat([added_entities, pd.DataFrame(data.query('is_nil')['Wikipedia_ID'].unique(), columns=['Wikipedia_ID'])]).drop_duplicates()
    added_entities.set_index('Wikipedia_ID', drop=False, inplace=True)

    # ## Entity Clustering
    print('clustering')
    nil_mentions = data.query('is_nil == True')

    res_nilcluster = requests.post(nilcluster, json={
            'ids': nil_mentions.index.tolist(),
            'mentions': nil_mentions[mention].values.tolist(),
            'encodings': nil_mentions['encoding'].values.tolist()
        })

    if not res_nilcluster.ok:
        print('NIL cluster ERROR')
    else:
        print('OK')

    clusters = pd.DataFrame(res_nilcluster.json())

    clusters = clusters.sort_values(by='nelements', ascending=False)

    ###### < eval clustering

    ## NIL clustering
    exploded_clusters = clusters.explode(['mentions_id', 'mentions'])
    merged = data.merge(exploded_clusters, left_index=True, right_on='mentions_id')

    # https://github.com/hhromic/python-bcubed
    keys = [str(x) for x in exploded_clusters['mentions_id']]
    values = [set([str(x)]) for x in exploded_clusters.index]
    cdict = dict(zip(keys, values))

    keysGold = [str(x) for x in merged['mentions_id']]
    # valuesGold= [set([x]) for x in merged['y_wikiurl_dump']]
    valuesGold= [set([x]) for x in merged['Wikipedia_ID']]
    ldict = dict(zip(keysGold, valuesGold))
    report['nil_clustering_bcubed_precision'] = bcubed.precision(cdict, ldict)
    report['nil_clustering_bcubed_recall'] = bcubed.recall(cdict, ldict)

    ###### eval clustering >

    if correct_steps:
        # get mentions list
        # get mentions ids
        # get embeddings and calculate medoid
        # get len of clusters
        def cluster_helper(x):
            cluster = {}
            cluster['mentions'] = x['mention'].tolist()
            cluster['mention_ids'] = x.index.tolist()
            cluster['title'] = pd.Series(cluster['mentions']).value_counts().index[0]
            cluster['center'] = KMedoids(n_clusters=1).fit(x['encoding']).cluster_centers_
            cluster['nelements'] = len(cluster['mentions'])
            
        clusters = nil_mentions.groupby('Wikipedia_ID').apply(cluster_helper)
        

    if not no_add:
        # populate with new entities
        print('Populating rw index with new entities')

        # inject url into cluster (most frequent one) # even NIL mentions still have the gold url
        def _helper(x):
            x = x['mentions_id']
            modes = data.loc[x, 'Wikipedia_ID'].mode()
            if len(modes) > 1:
                mode = None
            else:
                mode = modes[0]
            return {'mode': mode, 'modes': modes}

        clusters[['mode', 'modes']] = clusters.apply(_helper, axis=1, result_type='expand')

        data_new = clusters[['title', 'center']].rename(columns={'center': 'encoding', 'mode': 'wikipedia_id'})
        new_indexed = requests.post(indexer_add, json=data_new.to_dict(orient='records'))

        if not new_indexed.ok:
            print('error adding new entities')
        else:
            new_indexed = new_indexed.json()
            clusters['index_id'] = new_indexed['ids']
            clusters['index_indexer'] = new_indexed['indexer']
            prev_clusters = prev_clusters.append(clusters[['index_id', 'mode', 'modes']], ignore_index=True)
            prev_clusters.set_index('index_id', drop=False, inplace=True)

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        batch_basename = os.path.splitext(os.path.basename(batch))[0]
        outdata = os.path.join(save_path, '{}_outdata.pickle'.format(batch_basename))
        data.to_pickle(outdata)
        outclusters = os.path.join(save_path, '{}_outclusters.pickle'.format(batch_basename))
        clusters.to_pickle(outclusters)


    if not correct_steps:
        # does overall results make sense if steps are correct?

        # Overall

        overall_correct = 0
        # not nil are corrected when linked to the correct entity and labeled as not-NIL
        overall_to_link_correct = data.eval('~NIL and ~is_nil and Wikipedia_ID == top_Wikipedia_ID').sum()
        report['overall_to_link_correct'] = overall_to_link_correct / (data.eval('~NIL').sum() + sys.float_info.min)
        overall_correct += overall_to_link_correct
        # nil not yet added to the kb are correct if labeled as NIL
        should_be_nil = data[should_be_nil_bool]
        report['should_be_nil_correct'] = should_be_nil.eval('is_nil').sum()
        report['should_be_nil_total'] = should_be_nil.shape[0]
        report['should_be_nil_correct_normalized'] = report['should_be_nil_correct'] / (report['should_be_nil_total'] + sys.float_info.min)
        overall_correct += report['should_be_nil_correct']
        # nil previously added should be linked to the prev added entity (and not nil)
        should_be_linked_to_prev_added = data[data['NIL'] & data['Wikipedia_ID'].isin(prev_added_entities.index)]
        should_be_linked_to_prev_added_total = should_be_linked_to_prev_added.shape[0]
        should_be_linked_to_prev_added = should_be_linked_to_prev_added.query('~is_nil').copy()
        ## check if linked to a cluster containing at least half coherent mentions
        ### check if the Wikipedia_ID matches the majority of the Wikipedia_IDs in the cluster?
        should_be_linked_to_prev_added_correct = 0
        if should_be_linked_to_prev_added.shape[0] > 0:
            should_be_linked_to_prev_added[['top_candidate_id', 'top_candidate_indexer']] = \
                should_be_linked_to_prev_added.apply(\
                    lambda x: {
                        'top_candidate_id': x['candidates'][0]['id'],
                        'top_candidate_indexer': x['candidates'][0]['indexer']
                    }, axis=1, result_type = 'expand')

            index_indexer = clusters.iloc[0]['index_indexer']
            assert clusters.eval(f'index_indexer == {index_indexer}').all()

            # filter only the ones with the correct indexer
            should_be_linked_to_prev_added = should_be_linked_to_prev_added.query(f'top_candidate_indexer == {index_indexer}')

            # check they are linked correctly
            should_be_linked_to_prev_added = should_be_linked_to_prev_added.merge(prev_clusters, left_on = 'top_candidate_id', right_index = True)

            if should_be_linked_to_prev_added.shape[0] > 0:
                # the majority of the cluster is correct
                should_be_linked_to_prev_added_correct = should_be_linked_to_prev_added.eval('Wikipedia_ID == mode').sum()
                # half of the cluster is correct
                def helper_half_correct(row):
                    return len(row['modes']) == 2 and row['Wikipedia_ID'] in row['modes']
                should_be_linked_to_prev_added_correct += \
                    should_be_linked_to_prev_added.apply(helper_half_correct, axis=1).sum()
                overall_correct += should_be_linked_to_prev_added_correct

        report['should_be_linked_to_prev_added_correct'] = should_be_linked_to_prev_added_correct
        report['should_be_linked_to_prev_added_total'] = should_be_linked_to_prev_added_total
        report['should_be_linked_to_prev_added_correct_normalized'] = should_be_linked_to_prev_added_correct / (should_be_linked_to_prev_added_total + sys.float_info.min)

        report['overall_correct'] = overall_correct
        report['overall_accuracy'] = overall_correct / data.shape[0]

    ### print output
    pprint(report)

    return report

def explode_nil(row, column='nil_prediction', label=''):
    x = row[column]
    res = {}
    for k in x['True'].keys():
        res[f'NIL-{label}-{k}'] = x['True'][k]
    for k in x['False'].keys():
        res[f'not-NIL-{label}-{k}'] = x['False'][k]
    return res

@click.command()
@click.option('--no-add', is_flag=True, default=False, help='Do not add new entities to the KB.')
# @click.option('--cross', is_flag=True, default=False, help='Use also the crossencoder (implies --no-add).')
@click.option('--save-path', default=None, type=str, help='Folder in which to save data.')
@click.option('--no-reset', is_flag=True, default=False, help='Reset the RW index before starting.')
@click.option('--report', default=None, help='File in which to write the report in JSON.')
@click.option('--no-incremental', is_flag=True, default=False, help='Run the evaluation merging the batches')
@click.option('--prepare-for-nil-pred', is_flag=True, default=False, help='Prepare data for training NIL prediction. Combine with --save-path.')
@click.option('--correct-steps', is_flag=True, default=False, help='Evaluate every component supposing previous steps are correct.')
@click.argument('batches', nargs=-1, required=True)
def main(no_add, save_path, no_reset, report, batches, no_incremental, prepare_for_nil_pred, correct_steps):
    print('Batches', batches)
    outreports = []

    reset = not no_reset

    if prepare_for_nil_pred and (not save_path or report is not None):
        print('--prepare-for-nil-prediction requires --save-path and no --report')
        sys.exit(1)

    # check batch files exist
    for batch in batches:
        assert os.path.isfile(batch)

    # reset kbp
    if reset:
        print('Resetting RW index...')
        res_reset = requests.post(indexer_reset, data={})

        if res_reset.ok:
            print('Reset done.')
        else:
            print('ERROR while resetting!')
            sys.exit(1)

    if no_incremental:
        print('*** NO INCREMENTAL ***')
        print('Loading and combining batches')
        datas = list(map(lambda x: pd.read_json(x, lines=True), batches))
        data = pd.concat(datas, ignore_index=True)
        outreport = run_batch("no_incremental", data, no_add, save_path, prepare_for_nil_pred)
        outreports.append(outreport)
    else:
        for batch in tqdm(batches):
            print('Loading batch...', batch)
            data = pd.read_json(batch, lines=True)
            outreport = run_batch(batch, data, no_add, save_path, prepare_for_nil_pred, correct_steps)
            outreports.append(outreport)

    if report:
        report_df = pd.DataFrame(outreports)
        temp_df = report_df.apply(lambda x: explode_nil(x, 'nil_prediction'), result_type='expand', axis=1)
        report_df[temp_df.columns] = temp_df
        temp_df = report_df.apply(lambda x: explode_nil(x, 'nil_prediction_mitigated', 'mitigated'), result_type='expand', axis=1)
        report_df[temp_df.columns] = temp_df
        report_df = report_df.drop(columns=['nil_prediction', 'nil_prediction_mitigated'])
        if not no_incremental:
            incremental_overall = report_df.mean(numeric_only=True)
            incremental_overall['batch'] = 'incremental_overall'
            incremental_overall['overall_correct'] = report_df['overall_correct'].sum()
            incremental_overall['size'] = report_df['size'].sum()
            incremental_overall['overall_accuracy'] = incremental_overall['overall_correct'] / incremental_overall['size']
            incremental_overall

            report_df = report_df.append(incremental_overall, ignore_index=True)

        report_df.to_csv(report)

if __name__ == '__main__':
    main()
