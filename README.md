# Disclaimer

Tested on GNU/Linux.
Running the pipeline requires about:
- 30G RAM (the index of 6M wikipedia entities is kept in memory)
- 50G disk (index, entities information, models)
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
Download from [here](https://drive.google.com/drive/folders/1QmLhKpVwG_s9NVawsTpwrSB2sbdsPI9W?usp=sharing) and extract.

## Create the dataset
Follow the notebook [create_dataset.ipynb]()

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
TODO notebook with example text

## Evaluate
TODO run script for evaluation and explain the metrics

## Train NIL prediction
TODO run script to train NIL prediction models

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