import os
import wandb
import pandas as pd
import warnings
import torch
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
from collections import defaultdict
from datasets import Dataset
from scipy.stats import pearsonr
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments
from sklearn.model_selection import train_test_split
from transformers import (
    Trainer,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"


def modify_classification_head(model):
    model.classifier = nn.Sequential(
        model.classifier,
        nn.Tanh(),
    )
    return model


def create_dataset(df):
    data = defaultdict(list)

    for _, row in tqdm(df.iterrows(), total=len(df)):
        row = row.to_dict()
        data["input_ids"].append(row["input_ids"])
        data["attention_mask"].append(row["attention_mask"])
        data["labels"].append(row["score"])

    return Dataset.from_pandas(pd.DataFrame(data))


def main(**kwargs):
    tokenizer = AutoTokenizer.from_pretrained(kwargs["model_name"], use_fast=True)

    model = modify_classification_head(
        AutoModelForSequenceClassification.from_pretrained(
            kwargs["model_name"], num_labels=1
        )
    )

    lora_config = LoraConfig(
        task_type="SEQ_CLS",
        bias="none",
        inference_mode=False,
        r=kwargs["lora_r"],
        lora_alpha=kwargs["lora_alpha"],
        lora_dropout=kwargs["lora_dropout"],
        target_modules=kwargs["lora_module"].split(","),
        modules_to_save=["classifier"],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=kwargs["output_dir"],
        logging_dir=None,
        overwrite_output_dir=True,
        num_train_epochs=kwargs["epochs"],
        per_device_train_batch_size=kwargs["batch_size"],
        per_device_eval_batch_size=kwargs["batch_size"],
        evaluation_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=5,
        eval_steps=10,
        save_total_limit=1,
        bf16=True,
        seed=42,
        save_steps=10,
        dataloader_num_workers=20,
        lr_scheduler_type=kwargs["lr_scheduler"],
        warmup_steps=kwargs["warmup_steps"],
        optim=kwargs["optimizer"],
        learning_rate=kwargs["lr"],
        report_to="wandb",
    )

    df = pd.read_csv("text_with_scores_combined.csv")

    tokenized_texts = tokenizer(
        df["text"].tolist(),
        truncation=True,
        padding=True,
        return_tensors="pt",
        max_length=200,
    )

    df["input_ids"] = tokenized_texts["input_ids"].tolist()
    df["attention_mask"] = tokenized_texts["attention_mask"].tolist()
    df["score_binned"] = pd.cut(df["score"], bins=10, labels=False)

    train_df, val_df = train_test_split(
        df, test_size=0.1, random_state=123, stratify=df["score_binned"], shuffle=True
    )

    train_dataset = create_dataset(train_df)
    eval_dataset = create_dataset(val_df)
    print(f"Train dataset: {len(train_dataset)}")
    print(f"Eval dataset: {len(eval_dataset)}")
    print("=" * 100)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()

    final_eval_loss = trainer.evaluate(eval_dataset)["eval_loss"]
    wandb.log({"final_eval_loss": final_eval_loss})

    predictions_ = trainer.predict(eval_dataset)

    predictions = predictions_.predictions.flatten()
    true_labels = predictions_.label_ids

    final_pearson_corr, _ = pearsonr(predictions, true_labels)
    wandb.log({"final_pearson_corr": final_pearson_corr})

    torch.save(
        model.merge_and_unload(), os.path.join(kwargs["output_dir"], "best_model.pt")
    )

    del train_dataset, eval_dataset
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main(
        epochs=30,
        lr=1e-3,
        warmup_steps=40,
        batch_size=16,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        lr_scheduler="linear",
        optimizer="adamw_torch",
        lora_module="key,value",
        output_dir="BID_model_new_v2",
        model_name="ai4bharat/IndicBERTv2-MLM-Sam-TLM",
    )
