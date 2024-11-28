#!/usr/bin/env python
# coding: utf-8

from transformers import AutoModel, AutoTokenizer, BertTokenizer, BertModel
import torch
import pandas as pd

THRESHOLD = 0.904
output_path = "tweets_selected_sentence_indic-bert_check.csv"
man_text = "आदमी"
woman_text = "महिला"
model_type = "indic"
DATA = "tweets" # (or kathbath)
agss_path = "AGSS-zh-CN-hi.xlsx"
eng_hi_adj_path = "eng-hi_adj.csv"
tweets_path = "corgi-tweets.json"
kathbath_prepped_path = "kathbath_hindi_prepped.tsv"


def preprocess_agss(agss):
    df = pd.read_excel(agss, header=None)
    hi_adj = df[0].to_list()
    return hi_adj

def calc_gender_diff():
    text = man_text
    encoded_input = tokenizer(text, return_tensors='pt')
    man = model(**encoded_input)

    text = woman_text
    encoded_input = tokenizer(text, return_tensors='pt')
    woman = model(**encoded_input)

    diff = torch.sub(man[1],woman[1])
    return diff


if model_type == "indic":
    tokenizer = AutoTokenizer.from_pretrained('ai4bharat/indic-bert')
    model = AutoModel.from_pretrained('ai4bharat/indic-bert')
else:
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    model = BertModel.from_pretrained("bert-base-multilingual-cased")

df = pd.read_csv(eng_hi_adj_path)
hindi_adj = df["words"].to_list()

processed_agss = preprocess_agss(agss_path)
hindi_adj.extend(processed_agss)

gender_diff = calc_gender_diff()

adj_dict = {}
for i in hindi_adj:
    encoded_input = tokenizer(i, return_tensors='pt')
    adj = model(**encoded_input)
    dot_p = torch.dot(torch.flatten(adj[1]),torch.flatten(gender_diff))
    adj_dict[i] = dot_p

fem_dict = {}
male_dict = {}

for i in adj_dict:
    if adj_dict[i] < 0:
        fem_dict[i] = adj_dict[i]
    else:
        male_dict[i] = adj_dict[i]

conv_fem_dict = {k : abs(v) for k, v in fem_dict.items()}

max_val = max(conv_fem_dict.values())
norm_fem_dict = {k : (v / max_val) for k, v in conv_fem_dict.items()}
norm_fem_dict = dict(sorted(norm_fem_dict.items(), key=lambda item: item[1]))

final_fem_dict = {}
for i in norm_fem_dict:
    if norm_fem_dict[i] > THRESHOLD: # play with the threshold -> mbert, 0.33 (tweets), 0.45 (kathbath), indic-bert 0.904 (kathbath, tweets), 0.902 (emohi) 
        final_fem_dict[i] = norm_fem_dict[i]

if DATA == "tweets":
    jsonObj = pd.read_json(path_or_buf = tweets_path, lines=True)
    txt = jsonObj["tweet"]
else:
    df = pd.read_csv(kathbath_prepped_path, sep="\t")
    txt = df["text"].to_list()

word_dict = {}
sen_dict = {}

for n, line in enumerate(txt):
    sen_dict[n] = line
    split_line = line.split()
    for word in split_line:
        if word in word_dict:
            word_dict[word].append(n)
        else:
            word_dict[word] = [n]

sen_no = []
for i in final_fem_dict:
    if i in word_dict:
        sen_no.extend(word_dict[i])

unique_sen_no = list(set(sen_no))

selected_sentences = []

for i in list(unique_sen_no)[:100]:
    selected_sentences.append((sen_dict[i]))
    
new_df = pd.DataFrame()
new_df["selected_sentences"] = selected_sentences
new_df.to_csv(output_path, index=False)