import torch
import numpy as np
import pandas as pd
from transformers import BartTokenizer
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling
import logging

from config import Config
from datasets import Dataset
from retro_pytorch import RETRO, TrainingWrapper


# Set log level
logging.basicConfig(
    level=logging.DEBUG,
    format="(%(asctime)s) | %(levelname)-8s | %(module)s: %(message)s",
    datefmt='%Y-%m-%d %H:%M:%S')


class TrainPipeline:
    def __init__(self, config):
        self.config = config
        # Initialize retro transformer
        retro = RETRO(
            max_seq_len=2048,                      # max sequence length
            enc_dim=896,                           # encoder model dimension
            enc_depth=3,                           # encoder depth
            dec_dim=768,                           # decoder model dimensions
            dec_depth=12,                          # decoder depth
            dec_cross_attn_layers=(1, 3, 6, 9),    # decoder cross attention layers (with causal chunk cross attention)
            heads=8,                               # attention heads
            dim_head=64,                           # dimension per head
            dec_attn_dropout=0.25,                 # decoder attention dropout
            dec_ff_dropout=0.25  ,
            chunk_size=128                  # decoder feedforward dropout
        ).cuda()

        # Initialize retro's retrieval database
        wrapper = TrainingWrapper(
            retro=retro,                                 # path to retro instance
            knn=2,                                       # knn (2 in paper was sufficient)
            chunk_size=128,                               # chunk size (64 in paper)
            documents_path='/Python/shihao/RETRO-pytorch/data',
            glob='**/*.csv',
            chunks_memmap_path='./train.chunks.dat',     # path to chunks
            seqs_memmap_path='./train.seq.dat',          # path to sequence data
            doc_ids_memmap_path = './train.doc_ids.dat',   # path to document ids per chunk (used for filtering neighbors belonging to same document)
            max_chunks=1_000_000,                        # maximum cap to chunks
            max_seqs=100_000,                            # maximum seqs
            knn_extra_neighbors=100,                     # num extra neighbors to fetch
            max_index_memory_usage='100m',
            current_memory_available='1G'
        )

        # Convert retro token ids to text
        dataloaders = iter(wrapper.get_dataloader(batch_size=config['batch_size'], shuffle=True))
        # Bert tokenizer
        bert_tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-cased')
        # Convert bert token ids into words
        bart_inputs = []
        for seq, retrieved in dataloaders:
            retrieved = retrieved.reshape(retrieved.size(0), -1)
            inputs = torch.cat([seq, retrieved], -1)

            for input in inputs:
                bart_input = bert_tokenizer.convert_ids_to_tokens(input)
                bart_inputs.append(bart_input)
        bart_inputs = np.stack(bart_inputs, 0)
        bart_inputs = list(map(lambda x: ' '.join(x), bart_inputs))
        bart_inputs = pd.DataFrame({'text': bart_inputs})
        self.dataset = Dataset.from_pandas(bart_inputs)
        
        # Train-test split
        self.dataset = self.dataset.train_test_split(test_size=config['test_size'])

        # Load tokenizer
        self.tokenizer = BartTokenizer.from_pretrained(config['model']['name'])
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Create data collator
        self.data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=config['model']['name'], **config['model']['args'])

        # Load training arguments
        self.training_args = TrainingArguments(**self.config['training_args'])

    def _preprocess(self, examples):
        return self.tokenizer(examples['text'], truncation=True)

    def _group_texts(self, examples):
        # flatten all
        block_size = 128
        concatenated_examples = dict()
        for k in examples.keys():
            concatenated_examples[k] = sum(examples[k], [])
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()

        return result

    def train(self):
        # Preprocess dataset
        tokenized_dataset = self.dataset.map(
            self._preprocess, batched=True, num_proc=8,
            remove_columns=self.dataset["train"].column_names,
        )
        lm_dataset = tokenized_dataset.map(self._group_texts, batched=True, num_proc=8)

        # Configure trainer
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=lm_dataset['train'],
            eval_dataset=lm_dataset['test'],
            data_collator=self.data_collator,
        )

        # Train
        train_result = self.trainer.train()

        # Save tokenizer and model
        self.trainer.save_model()
        # Save trainer state
        self.trainer.save_state()

        # Save metrics
        metrics = train_result.metrics
        self.trainer.log_metrics("train", metrics)
        self.trainer.save_metrics("train", metrics)
        
        return None


if __name__ == '__main__':
    TrainPipeline(Config.train).train()
