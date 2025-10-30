from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, AutoTokenizer
import torch
import wandb
from dataset_processor import DatasetProcessor
from typing import List, Dict
from pathlib import Path
import time
from omegaconf import OmegaConf
import os
import gc
dtype = torch.bfloat16

# ----- CONFIG ------ #

def load_config(config_path: str = './config/experiments.yaml'):
    """Load configuration from a YAML file using OmegaConf.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        Any: The loaded OmegaConf DictConfig.
    """
    resolved_path = os.path.abspath(config_path)
    print(f'📁 CONFIG: Loading configuration from {resolved_path}')
    if not os.path.exists(resolved_path):
        raise FileNotFoundError(f"Config file not found: {resolved_path}")
    config = OmegaConf.load(resolved_path)
    print(f'✅ CONFIG: Successfully loaded configuration with {len(config.experiments)} experiments ✅')
    return config

cfg = load_config()
# ----- DATASET ----- #

dataset_ = DatasetProcessor(tokenizer_name = cfg.base_model, n_shards_per_dataset=5)
train_dataset = dataset_()
train_dataset = train_dataset.shuffle()
print(train_dataset)
time.sleep(10)


# ----- MODEL TRAIN ----- #

class TTSPadCollator:
    def __init__(self, pad_token_id: int, label_pad_id: int = -100):
        self.pad_token_id = pad_token_id
        self.label_pad_id = label_pad_id

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)

        input_ids = []
        attention_masks = []
        labels = []

        for item in features:
            pad_len = max_len - len(item["input_ids"])

            if pad_len > 0:
                item["input_ids"] = item["input_ids"] + [self.pad_token_id] * pad_len
                item["attention_mask"] = item["attention_mask"] + [0] * pad_len
                item["labels"] = item["labels"] + [self.label_pad_id] * pad_len

            input_ids.append(item["input_ids"])
            attention_masks.append(item["attention_mask"])
            labels.append(item["labels"])

        batch = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
        return batch


class ItemTrain:
    def __init__(self, base_model_name:str, project_name:str, experiment_cfg:OmegaConf)->None:
        self.base_model_name = base_model_name
        self.project_name = project_name
        self.cfg = experiment_cfg
        self.model_id = self.cfg.base.model_id
        self.run_name = self.cfg.base.run_name
        checkpoint_dir = getattr(self.cfg.base, "checkpoint_dir", '/scratch2/kani_tts/au_500k/checkpoints')
        self.base_repo_path = Path(checkpoint_dir).expanduser().resolve()
        self.base_repo_path.mkdir(parents=True, exist_ok=True)

        self.lora_config = LoraConfig(**self.cfg.lora_args)
        self.training_args = TrainingArguments(**self.cfg.trainer_args,
                                        overwrite_output_dir=True,
                                        logging_steps=1,
                                        output_dir=str(self.base_repo_path),
                                        report_to="wandb",
                                        save_strategy="no",
                                        remove_unused_columns=True,
                                        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            attn_implementation="flash_attention_2",
            dtype=dtype,
            trust_remote_code=True,
        )
        self.model.config.use_cache = False
        self.model = get_peft_model(self.model , self.lora_config)
        self.data_collator = TTSPadCollator(pad_token_id=self.tokenizer.pad_token_id)

    def __call__(self)->None:
        print(f"=== TRAIN THE {self.model_id} MODEL ===")
        wandb.init(project=self.project_name, name = self.run_name)

        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            data_collator=self.data_collator,
        )

        trainer.train()

        merged_model = self.model.merge_and_unload()
        save_dir = self.base_repo_path / self.model_id
        merged_model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        wandb.finish()

# ----- Experiments ----- #
for item_cfg in cfg.experiments:
    experiment = None

    try:
        experiment = ItemTrain(base_model_name = cfg.base_model,
                                project_name = cfg.project_name,
                                experiment_cfg = item_cfg)
        experiment()

    except Exception as e:
        print(f'ERROR WITH {item_cfg.base.model_id}: {e}')

    finally:
        if experiment:
            del experiment.model
            del experiment.tokenizer
            del experiment
        torch.cuda.empty_cache()
        gc.collect()
        print(f"VRAM cleared. Available: {torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()}")
    
