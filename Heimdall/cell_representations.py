"""
The Cell Representation Object for Processing

"""

import scanpy as sc
import anndata as ad
import numpy as np
# import torch
from Heimdall.utils import get_value



class Cell_Representation:
    def __init__(self, config):
        """
        Initialize the Cell Rep object with configuration and AnnData object.

        Parameters:
        config (dict): Configuration dictionary.
        """
        self.dataset_preproc_cfg = config.dataset.preprocess_args
        self.dataset_task_cfg = config.dataset.task_args
        self.fg_cfg = config.f_g
        self.fc_cfg = config.f_c
        self.model_cfg = config.model
        self.optimizer_cfg = config.optimizer
        self.trainer_cfg = config.trainer
        self.scheduler_cfg = config.scheduler
        self.adata = None

    
    def preprocess_anndata(self):
        if self.adata is not None:
            raise ValueError("Anndata object already exists, are you sure you want to reprocess again?")

        # Load your AnnData object
        self.adata = ad.read_h5ad(self.dataset_preproc_cfg.data_path)
        print(f"> Finished Loading in {self.dataset_preproc_cfg.data_path}")

        if(get_value(self.dataset_preproc_cfg, "normalize")):
            # Normalizing based on target sum
            print("> Normalizing anndata...")
            sc.pp.normalize_total(self.adata, target_sum=1e4)
            assert self.dataset_preproc_cfg.normalize and self.dataset_preproc_cfg.log_1p, "Normalize and Log1P both need to be TRUE"
        else:
            print("> Skipping Normalizing anndata...")

        if(get_value(self.dataset_preproc_cfg, "log_1p")):
            ## log Transform step
            print("> Log Transforming anndata...")
            sc.pp.log1p(self.adata)
        else:
            print("> Skipping Log Transforming anndata..")

        if(get_value(self.dataset_preproc_cfg, "top_n_genes")):
            # Identify highly variable genes
            print(f"> Using highly variable subset... top {self.dataset_preproc_cfg.top_n_genes} genes")
            sc.pp.highly_variable_genes(self.adata, n_top_genes=self.dataset_preproc_cfg.top_n_genes)
            self.adata = self.adata[:, self.adata.var['highly_variable']]
        else:
            print(f"> No highly variable subset... using entire dataset")
        
        if(get_value(self.dataset_preproc_cfg, "scale_data")):
            # Scale the data
            print("> Scaling the data...")
            sc.pp.scale(self.adata, max_value=10)
        else:
            print("> Not Scaling the data...")

        print("> Finished Processing Anndata Object")




    def prepare_labels(self):
        """
        Prepares the self.labels by pulling out the specified class from 
        """
        assert self.adata is not None, "no adata found, Make sure to run preprocess_anndata() first"

        df = self.adata.obs
        class_mapping = {label: idx for idx, label in enumerate(df[self.dataset_task_cfg.label_col_name].unique(), start=0)}
        df['class_id'] = df[self.dataset_task_cfg.label_col_name].map(class_mapping)
        self.labels = np.array(df["class_id"])
        print(f"> Finished extracting labels, self.labels.shape: {self.labels.shape}")



    def preprocess_f_g(self, f_g):
        """
        run the f_g, and then preprocess and store it locally
        the f_g must return a `gene_mapping` where the keys are the gene ids
        and the ids are the.

        The f_g will take in the anndata as an object just for flexibility

        For example:
        ```
        {'Sox17': 0,
        'Rgs20': 1,
        'St18': 2,
        'Cpa6': 3,
        'Prex2': 4,
        ...
        }

        or even: 
        {'Sox17': [0.3, 0.1. 0.2 ...],
        'Rgs20':  [0.3, 0.1. 0.2 ...],
        'St18':  [0.3, 0.1. 0.2 ...],
        'Cpa6':  [0.3, 0.1. 0.2 ...],
        'Prex2':  [0.3, 0.1. 0.2 ...],
        ...
        }
        ```
        """
        assert self.fg_cfg.args.output_type in ["ids", "vector"]

        ## For identity, we convert each of the vars into a unique ID
        gene_mapping = f_g(self.adata.var)

        assert all(isinstance(value, (np.ndarray, list, int)) for value in gene_mapping.values()), \
            "Make sure that all values in the gene_mapping dictionary are either int, list or np array"

        self.f_g = gene_mapping
        print(f"> Finished calculating f_g with {self.fg_cfg.name}")
        return



    def preprocess_f_c(self, f_c):
        """
        Preprocess the cell f_c, this will preprocess the anndata.X into the actual dataset to 
        the actual tokenizers, you can imagine this as a cell tokenizer.

        The f_c will take as input the f_g, then the anndata, then the 
        """
        self.cell_representation = f_c(self.f_g, self.adata)
        print(f"> Finished calculating f_c with {self.fc_cfg.name}")
        return