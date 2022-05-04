# Disclaimer

Tested on GNU/Linux.
Running the pipeline requires about:
- ~33G RAM (the index of 6M wikipedia entities is kept in memory)
- ~60G disk (index, entities information, models)
- GPU is not mandatory but recommended (at least for running the pipeline on an entire dataset)

# Project structure
```
notebooks           # interactive notebooks
pipeline            # pipeline services
README.md
requirements.txt    # python requirements for notebooks and scripts
scripts             # scripts
```
TODO: go deeper

# Python Environment
To run the notebooks and the scripts in this repo you can use a python3 environment (See https://docs.python.org/3/library/venv.html).

## Create the environment
Tested on Python 3.8.10.
```
python3 -m venv venv
source venv/bin/activate # works with bash
```

## Install the requirements
```
pip install --upgrade pip # upgrade pip
pip install -r requirements.txt
```

# Dataset
Download the dataset or create it starting from Wikilinks Unseen-Mentions.

## Download the dataset
Download from [here](https://drive.google.com/drive/folders/1QmLhKpVwG_s9NVawsTpwrSB2sbdsPI9W?usp=sharing) and extract it into the main folder of the project. You should see something like this:
```
.
├── incremental_dataset
│   ├── delete_nil_entities.sql
│   ├── dev
│   ├── statistics
│   ├── test
│   └── train
├── notebooks
│   ├── create_dataset.ipynb
├── pipeline
├── README.md
├── requirements.txt
└── scripts
```

## Create the dataset
Follow the notebook [create_dataset.ipynb]() and then copy the datset folder in the root directory of the project as shown in the previous directory structure.

# Pipeline
Every component of the pipeline is deployed as a microservice exposing HTTP APIs.
The services are:
- biencoder: uses the biencoder to encode mentions (or entities) into vectors.
- indexer: given a vector it runs the (approximate) nearest neighbor algorithm to retrieve the best candidates for linking .
- nilpredictor: given mention and the best candidates if estimates wheter the mention is NIL or the linking is correct.
- nilcluster: given a set of NIL mentions if cluster together the ones referring to the same (not in the KB) entity.

The pipeline requires the following additional services:
- postgres database: it keeps the information about the entities (to avoid keeping them in memory).

## Setup with Docker
Docker (possibly with GPU support) and Compose are required.
Follow these links to install them.
- Docker: https://docs.docker.com/get-docker/
- Compose: https://docs.docker.com/compose/install/
- Nvidia Docker: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#setting-up-nvidia-container-toolkit

## Prepare
Enter the pipeline folder then create a folder named `models` in the root folder of the project (same folder of `docker-compose.yml`), if it does not exist.

### Download models
We need to download these files and put them in the models directory:
- the biencoder model (from Meta Research):
    - http://dl.fbaipublicfiles.com/BLINK/biencoder_wiki_large.bin
    - http://dl.fbaipublicfiles.com/BLINK/biencoder_wiki_large.json
- the index (from Meta Research):
    - http://dl.fbaipublicfiles.com/BLINK/faiss_hnsw_index.pkl
- the information about the entities in the index (from Meta Research):
    - http://dl.fbaipublicfiles.com/BLINK/entity.jsonl
    - TODO ensure populate works if not gzipped
- the NIL prediction model:
    - the file `nilp_bi_max_levenshtein_jaccard_model.pickle` from [here](https://drive.google.com/drive/folders/1QmLhKpVwG_s9NVawsTpwrSB2sbdsPI9W?usp=sharing)

Once downloaded the model folder should look like this:
```
pipeline/models/
pipeline/models/biencoder_wiki_large.bin                        (2.5G)
pipeline/models/biencoder_wiki_large.json
pipeline/models/entity.jsonl                                    (3.2G)
pipeline/models/faiss_hnsw_index.pkl                            (29G)
pipeline/models/nilp_bi_max_levenshtein_jaccard_model.pickle
```

### Prepare environment variables
Go back to the `pipeline` folder and copy the file `env-sample.txt` to `.env`, then edit the latter so that it fits your needs.

### Populate entity database
We need to populate the database with entities information (e.g. Wikipedia IDs, titles).

From inside the `pipeline` folder start postgres by running
```
# you may need to use sudo
docker-compose up -d postgres
```
Now postgres should listen on `tcp://127.0.0.1:5432`.

Let postgres run some seconds to initialize itself, then go back to the main directory of the project and with the python environment activated run the population script as follows. In case you changed the postgres password in the `.env` file replace `secret` in the following command with the password you chose.
```
python scripts/postgres_populate_entities.py --postgres postgresql://postgres:secret@127.0.0.1:5432/postgres --table-name entities --entity_catalogue pipeline/models/entity.jsonl --indexer 10
```

At this point you can delete `pipeline/models/entity.jsonl` since the information is in the database.

#### Delete NIL entities from the database
If you created a different dataset (by changing random seeds) you sould use the new sql command created by the notebook.

Run the sql query from the file `incremental_dataset/delete_nil_entities.sql`: you could use the following command from the pipeline folder:
```
# you may need sudo
docker-compose exec -T postgres psql -U postgres < ../incremental_dataset/delete_nil_entities.sql
```

### Enable GPU

In case you want to disable the GPU see [here](Without GPU).

Otherwise ensure GPU is enabled or enable it by editing the JSON file `models/biencoder_wiki_large.json` setting
```
no_cuda: false
```

### Start the services
From inside the pipeline folder run
```
# you may need sudo
docker-compose up -d
```

## Try the pipeline
Use the notebook [try_pipeline](notebooks/try_pipeline.ipynb).

## Evaluate
### Incremental Evaluation
Ensure the pipeline is up, then run:
```
python scripts/eval_kbp.py --report evaluation_report_incremental.csv \
    incremental_dataset/test/test_0.jsonl \
    incremental_dataset/test/test_1.jsonl \
    incremental_dataset/test/test_2.jsonl \
    incremental_dataset/test/test_3.jsonl \
    incremental_dataset/test/test_4.jsonl \
    incremental_dataset/test/test_5.jsonl \
    incremental_dataset/test/test_6.jsonl \
    incremental_dataset/test/test_7.jsonl \
    incremental_dataset/test/test_8.jsonl \
    incremental_dataset/test/test_9.jsonl
```

### One-pass Evaluation
Ensure the pipeline is up, then run:
```
python scripts/eval_kbp.py --no-incremental --report evaluation_report_onepass.csv \
    incremental_dataset/test/test_0.jsonl \
    incremental_dataset/test/test_1.jsonl \
    incremental_dataset/test/test_2.jsonl \
    incremental_dataset/test/test_3.jsonl \
    incremental_dataset/test/test_4.jsonl \
    incremental_dataset/test/test_5.jsonl \
    incremental_dataset/test/test_6.jsonl \
    incremental_dataset/test/test_7.jsonl \
    incremental_dataset/test/test_8.jsonl \
    incremental_dataset/test/test_9.jsonl
```

### Report
The report contains a line for each batch (also a line with the average over all the batches) with this metrics:
```
batch:                  batch identifier
size:                   batch size
linking_recall@1:       recall@k of the linking of not-NIL mentions
linking_recall@2:
linking_recall@3:
linking_recall@5:
linking_recall@10:
linking_recall@30:
linking_recall@100:
nil_prediction_cm:                  NIL prediction confusion matrix
nil_prediction_cm_normalized:       " normalized
nil_prediction_mitigated_cm:        NIL prediction mitigated (correct when a linking error is NIL)
nil_prediction_mitigated_cm_normalized:
nil_clustering_bcubed_precision:    NIL clustering bcubed precision
nil_clustering_bcubed_recall:       " recall
overall_to_link_correct:            linked_correcly / to_link
should_be_nil_correct:              number of correct nil
should_be_nil_total:                expected correct nil
should_be_nil_correct_normalized:   correct_nil / expected
should_be_linked_to_prev_added_correct:     number of mention correctly linked to entities added from previous clusters
should_be_linked_to_prev_added_total:       expected number of mentions to link to prev added entities
should_be_linked_to_prev_added_correct_normalized:  " normalized
overall_correct:        correct_predictions end-to-end
overall_accuracy:       " normalized
NIL--precision:         NIL prediction precision of the NIL class
NIL--recall:            " recall
NIL--f1-score:
NIL--support:
not-NIL--precision:     " of the not-NIL class
not-NIL--recall:
not-NIL--f1-score:
not-NIL--support:
NIL-mitigated-precision:    " mitigated (correct when a linking error is NIL)
NIL-mitigated-recall:
NIL-mitigated-f1-score:
NIL-mitigated-support:
not-NIL-mitigated-precision:
not-NIL-mitigated-recall:
not-NIL-mitigated-f1-score:
not-NIL-mitigated-support
```

## Train NIL prediction

In this example we train using the first batch of train from the incremental dataset and using the first batch of dev for evaluating and comparing the NIL prediction models.

Prepare data for the NIL prediction study/training: we need to get linking scores. Ensure the pipeline is up, then run
```
python scripts/eval_kbp.py --save-path output/prepare_for_nil_study --prepare-for-nil-pred --no-reset incremental_dataset/train/train_0.jsonl

python scripts/eval_kbp.py --save-path output/prepare_for_nil_study --prepare-for-nil-pred --no-reset incremental_dataset/dev/dev_0.jsonl
```
You should see two files in the folder `output/prepare_for_nil_study`:
```
train_0_outdata.pickle
dev_0_outdata.pickle
```

Then run the study and train the models with:
```
python scripts/feature_ablation_study.py --train-path output/prepare_for_nil_study/train_0_outdata.pickle --test-path output/prepare_for_nil_study/dev_0_outdata.pickle --output-path nilprediction_output
```

The `nilprediction_output` folder will contain:
- the models
- a summary (feature_ablation_summary.csv) that compares all the models
- plots of the distribution of the predictions
- performance report for each model
```
nilprediction_output/feature_ablation_summary.csv
nilprediction_output/nilp_bi_max_levenshtein_jaccard_model.pickle
nilprediction_output/nilp_bi_max_levenshtein_jaccard_kde_correct_errors.png
nilprediction_output/nilp_bi_max_levenshtein_jaccard_kde.png
nilprediction_output/nilp_bi_max_levenshtein_jaccard_report.txt
nilprediction_output/nilp_bi_max_levenshtein_jaccard_roc.png
...
```

## Without GPU
Edit the `docker-copmose.yml` commenting the part related to the gpu (Look for the comments in the file).
Edit the JSON file `models/biencoder_wiki_large.json` setting
```
no_cuda: true
```
We suggest to use GPU for evaluating a dataset, while to try the pipeline CPU should be enough.

# TODO
- ringraziamento simone repo
- citare facebook research (il codice arriva da li)
- cosa arriva in che cartella? incremental_dataset, models
- quali script servono ora?
    - populate postgres
    - train nil prediction
    - eval_kbp
        - refactor magari
- notebooks: creare cartella notebooks?
    - create dataset
    - try pipeline
- requirements degli scripts