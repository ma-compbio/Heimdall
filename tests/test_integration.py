import subprocess
from pathlib import Path


def test_train_geneformer_gene2vec_caching():
    subprocess.run(
        ["python", "train.py", "+experiments=classification_experiment_gene2vec_dev", "user=lane-shared-dev"],
    )


def test_train_geneformer_no_caching():
    subprocess.run(
        [
            "python",
            "train.py",
            "+experiments=classification_experiment_dev",
            "user=lane-shared-dev",
            "cache_preprocessed_dataset_dir=null",
        ],
    )
