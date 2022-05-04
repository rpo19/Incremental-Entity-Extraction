# Dataset
Download the dataset or create it starting from Wikilinks Unseen-Mentions

## Download the dataset
TODO prepare url (google drive?)

Download from here and extract.

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
Create a folder named `models` in the root folder of the project (same folder of `docker-compose.yml`), if it does not exist.

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
    - TODO link

Once downloaded the model folder should look like this:
```
models/
TODO...
```

### Populate entity database
We need to populate the database with entities information (e.g. Wikipedia IDs, titles).

Start postgres
```
# you may need superuser priviledges
docker-compose up -d postgres
```

Let postgres some seconds to setup, then run the population script:
```
# TODO virtualenv with requirements
python script/postgres_populate_entities.py --postgres postgresql://postgres:quauJae4eebeeleefie4han0shahreim@localhost:5432/postgres --table-name entities --entity_catalogue models/entity.jsonl --indexer 10
```

At this point you can delete `models/entity.jsonl` since the information is in the database.

### Start the services
Run
```
docker-compose up -d
```

## Try the pipeline
TODO notebook with example text

## Evaluate
TODO run script for evaluation and explain the metrics

## Train NIL prediction
TODO run script to train NIL prediction models

## Without GPU
Edit the JSON file `biencoder_wiki_large.json` setting
```
no_cuda: true
```
We suggest to use GPU for evaluating a dataset, while to try the pipeline CPU should be enough.