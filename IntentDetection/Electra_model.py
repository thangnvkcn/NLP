import pandas as pd
from sklearn.model_selection import train_test_split
import os
import torch
sep  = os.sep
data_folder = "Data"
file_path = '../intentDetection/Data/data_intent_17.csv'
NUM_WORDS = 1500
# def getData(file_path):
#     data = pd.read_csv(file_path)
#     return data
#
# data = getData(file_path)
# train, test = train_test_split(data, test_size=0.1,shuffle=True,random_state=123)
#
# train.to_csv(data_folder+sep+'train.csv', encoding='utf-8',index=False)
# test.to_csv(data_folder+sep+'test.csv', encoding='utf-8',index=False)

from simpletransformers.language_modeling import LanguageModelingModel
import logging
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

train_args = {
    "reprocess_input_data": False,
    "overwrite_output_dir": True,
    "num_train_epochs": 3,
    "save_eval_checkpoints": True,
    "save_model_every_epoch": False,
    "learning_rate": 5e-4,
    "warmup_steps": 10000,
    "train_batch_size": 64,
    "eval_batch_size": 128,
    "gradient_accumulation_steps": 1,
    "block_size": 128,
    "max_seq_length": 128,
    "dataset_type": "simple",
    "wandb_project": "Esperanto - ELECTRA",
    "wandb_kwargs": {"name": "Electra-SMALL"},
    "logging_steps": 100,
    "evaluate_during_training": True,
    "evaluate_during_training_steps": 50000,
    "evaluate_during_training_verbose": True,
    "use_cached_eval_features": True,
    "sliding_window": True,
    "vocab_size": 52000,
    "generator_config": {
        "embedding_size": 128,
        "hidden_size": 256,
        "num_hidden_layers": 3,
    },
    "discriminator_config": {
        "embedding_size": 128,
        "hidden_size": 256,
    },
}

train_file = "Data/train.csv"
test_file = "Data/test.csv"

model = LanguageModelingModel(
    "electra",
    None,
    args=train_args,
    train_files=train_file,
)


model.train_model(
    train_file, eval_file=test_file,
)

model.eval_model(test_file)