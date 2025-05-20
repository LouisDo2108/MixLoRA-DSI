import os
from copy import deepcopy
from dataclasses import asdict

import ujson
from t5_pretrainer.arguments import (
    Arguments,
    ModelArguments,
    TermEncoder_TrainingArguments,
)
from t5_pretrainer.dataset import MarginMSEDataset, T5DenseMarginMSECollator
from t5_pretrainer.ripor import T5DenseEncoderForMarginMSE
from t5_pretrainer.ripor_trainer import RiporTrainer
from t5_pretrainer.utils.utils import is_first_worker
from transformers import HfArgumentParser

local_rank = int(os.environ.get("LOCAL_RANK", 0))


def main():
    parser = HfArgumentParser((ModelArguments, Arguments))
    model_args, args = parser.parse_args_into_dataclasses() 

    # save args to disk
    if local_rank <= 0:
        merged_args = {**asdict(model_args), **asdict(args)}
        out_dir = deepcopy(args.output_dir)
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        with open(os.path.join(out_dir, "args.json"), "w") as f:
            ujson.dump(merged_args, f, indent=4)

    train_dataset = MarginMSEDataset(
        example_path=args.teacher_score_path,
        document_dir=args.collection_path,
        query_dir=args.queries_path
    )
    train_collator = T5DenseMarginMSECollator(
        tokenizer_type=model_args.model_name_or_path,
        max_length=args.max_length
    )

    model_args = None
    model = T5DenseEncoderForMarginMSE.from_pretrained(
        args.pretrained_path, model_args
    )  # For DocID Initialization

    evaluator = None

    training_args = TermEncoder_TrainingArguments(
        output_dir=args.output_dir,
        do_train=True,
        do_eval=args.do_eval,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        per_device_train_batch_size=args.per_device_train_batch_size,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        disable_tqdm=False,
        load_best_model_at_end=False,
        save_total_limit=1,
        remove_unused_columns=False,
        task_names=args.task_names,
        ln_to_weight=args.ln_to_weight,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        full_rank_eval_qrel_path=args.full_rank_eval_qrel_path,
        # My own parameters
        optim="adamw_8bit",
        dataloader_num_workers=4,
        gradient_accumulation_steps=1,
        tf32=True,
        bf16=True,
        seed=42,
        dataloader_pin_memory=True,
        report_to="none",
    )
    reg_to_reg_scheduler = None

    trainer = RiporTrainer(
        model=model,
        train_dataset=train_dataset,
        data_collator=train_collator,
        args=training_args,
        reg_to_reg_scheduler=reg_to_reg_scheduler,
        splade_evaluator=evaluator,
    )

    # Let's save the tokenizer first
    if is_first_worker():
        train_collator.tokenizer.save_pretrained(trainer.args.output_dir)
    trainer.train()
    trainer.save_torch_model_and_tokenizer(train_collator.tokenizer)


if __name__ == "__main__":
    main()
