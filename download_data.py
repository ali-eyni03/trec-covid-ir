from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

import logging
import pathlib, os

# Download dataset and unzip the dataset
dataset = "trec-covid"
url = (
    f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
)

out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "data", "raw")

data_path = util.download_and_unzip(url, out_dir)


# Delete zip file after extraction
zip_path = os.path.join(out_dir, f"{dataset}.zip")

if os.path.exists(zip_path):
    os.remove(zip_path)
    print(f"Deleted: {zip_path}")