import copy
import os
from pathlib import Path
from pprint import pprint

import msgspec
import torch
from safetensors import safe_open
from t5_pretrainer.dataset import RiporForSeq2seqCollator, RiporForSeq2seqDataset
from t5_pretrainer.mixlora import MixLoraDSI_naive_expand
from t5_pretrainer.mixlora_config import MixLoraConfig
from t5_pretrainer.mixlora_trainer import MixLoraDSI_Extendable_Trainer
from t5_pretrainer.mixlora_varigrow import MixLoraDSI_Varigrow
from t5_pretrainer.ripor import RiporForSeq2seq
from t5_pretrainer.utils.utils import is_first_worker, set_seed
from transformers import HfArgumentParser, Seq2SeqTrainingArguments
from transformers.utils import logging

torch.set_printoptions(sci_mode=False)

logger = logging.get_logger(__name__)

TRAINING_ARGS_NAME = "training_args.bin"
MODEL_ARGS_NAME = "model_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"

set_seed(42)


def get_argument_and_config():
    parser = HfArgumentParser(Seq2SeqTrainingArguments)  # type: ignore
    parser.add_argument("--model_name", default=None, type=str, required=True)
    parser.add_argument("--pretrained_path", default=None, type=str, required=True)
    parser.add_argument("--query_to_docid_path", default=None, type=str, required=True)
    parser.add_argument("--docid_to_smtid_path", default=None, type=str, required=True)
    parser.add_argument(
        "--mixlora_config_json_path",
        default="./MixLoraDSI/mixlora_config.json",
        type=str,
        required=True,
    )
    parser.add_argument("--taskid", type=int, default=0)

    # Only parse to HF's TrainingArguments valid keys
    training_args = parser.parse_args_into_dataclasses()[0]

    # Full arguments with added arguments
    args = parser.parse_args()

    # MixLoRA config
    with open(args.mixlora_config_json_path, "rb") as f:
        mixlora_config = msgspec.json.Decoder().decode(f.read())

    mixlora_config = MixLoraConfig.from_config(mixlora_config)

    return training_args, args, mixlora_config


def get_dataset_and_data_collator(args):
    # Dataset
    train_dataset = RiporForSeq2seqDataset(
        example_path=args.query_to_docid_path,
        docid_to_smtid_path=args.docid_to_smtid_path,
    )

    # Data collator
    data_collator = RiporForSeq2seqCollator(tokenizer_type="t5-base", max_length=256)
    return train_dataset, data_collator


def get_model(args, mixlora_config):
    model = MixLoraDSI_naive_expand.from_pretrained(
        model_name_or_path=args.pretrained_path,
        mixlora_config=mixlora_config,
    )

    checkpoint = {}
    if (Path(args.pretrained_path) / "model.safetensors").exists():
        with safe_open(Path(args.pretrained_path) / "model.safetensors", framework="pt", device="cpu") as f:  # type: ignore
            # This is most likely t5-self-neg checkpoint
            for k in f.keys():
                checkpoint[k] = f.get_tensor(k)
    elif (Path(args.pretrained_path) / "pytorch_model.bin").exists():
        checkpoint = torch.load(
            os.path.join(args.pretrained_path, "pytorch_model.bin"), map_location="cpu"
        )

    return model, checkpoint


def main():
    training_args, args, mixlora_config = get_argument_and_config()

    train_dataset, data_collator = get_dataset_and_data_collator(args)

    """
    Before training the model, we need to:
    1. Freeze the base model
    2. Extend the model with new experts and router weights
    """
    model, _ = get_model(args, mixlora_config)
    model._before_task(taskid=args.taskid, freeze_vocab=mixlora_config.freeze_vocab)

    if mixlora_config.kl_loss:
        model.previous_checkpoint = copy.deepcopy(model.base_model).cuda()

    trainer = MixLoraDSI_Extendable_Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        freeze_modules=None,
        mixlora_config=mixlora_config,
    )

    print(" ***************** MixLoRA config ***************** ")
    pprint(model.mixlora_config)
    print(" ***************** MixLoRA config ***************** ")

    if is_first_worker():
        data_collator.tokenizer.save_pretrained(trainer.args.output_dir)
    os.makedirs(
        os.path.join("./MixLoraDSI/logs", args.run_name),
        exist_ok=True,
    )
    # Clear everything in the log directory
    os.system(f"rm -rf ./MixLoraDSI/logs/{args.run_name}/*")

    trainer.train()
    trainer.save_torch_model_and_tokenizer(data_collator.tokenizer)
    # Move the model from checkpoint-* directory to the output directory
    os.system(
        f"mv {trainer.args.output_dir}/checkpoint-*/* {trainer.args.output_dir}"
    )


if __name__ == "__main__":
    main()
