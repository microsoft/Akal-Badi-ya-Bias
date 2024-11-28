import os
import re
import pandas as pd
import warnings
import torch
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
from transformers import AutoTokenizer

warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
device = "cuda" if torch.cuda.is_available() else "cpu"


def modify_classification_head(model):
    model.classifier = nn.Sequential(
        model.classifier,
        nn.Tanh(),
    )
    return model


def clean(x):
    x = str(x)
    x = (
        x.strip()
        .replace("&#39;", "'")
        .replace("&quot;", "`")
        .replace("&amp;", "&")
        .replace("’", "'")
        .replace("<br>", " ")
    )
    x = re.sub(r"\s+", " ", x)
    return x


def main(**kwargs):
    tokenizer = AutoTokenizer.from_pretrained(kwargs["model_name"], use_fast=False)
    model = torch.load(kwargs["ckpt"]).to(device)

    breakpoint()

    df = pd.read_csv(kwargs["infname"], encoding="utf-8").map(clean)
    df = df[df["comments"].notna()]
    sentences = df["comments"].tolist()
    scores = []

    for i in tqdm(range(0, len(df), 16)):
        batch = sentences[i : i + 16]
        inputs = tokenizer(
            text=batch,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
            return_attention_mask=True,
        ).to(device)

        with torch.inference_mode():
            outputs = model(**inputs)
            scores.extend(outputs.logits.squeeze(-1).tolist())

    df["score"] = scores
    df.to_excel(kwargs["outfname"], index=False)


if __name__ == "__main__":
    main(
        ckpt="BID_model_new_v2/best_model.pt",
        infname="youtube_full.csv",
        outfname="youtube_full_with_scores.xlsx",
        model_name="ai4bharat/IndicBERTv2-MLM-Sam-TLM",
    )
