"""
Script for creating sentence-embeddings from raw sentences.
INPUT: {lang}.txt
OUTPUT: {lang}.hdf5
"""

import gc
import torch
import argparse
import numpy as np
from time import time
from itertools import takewhile, repeat
from sentence_transformers import SentenceTransformer


device = "cuda" if torch.cuda.is_available() else "cpu"


def count_lines(filename):
    f = open(filename, "rb")
    bufgen = takewhile(lambda x: x, (f.raw.read(1024 * 1024) for _ in repeat(None)))
    return sum(buf.count(b"\n") for buf in bufgen)


def get_line(f_path):
    with open(f_path) as file:
        for line in file:
            yield line


def compute_embeddings(model, sentences, batch_size=1024, device="cuda"):
    while batch_size > 1:
        try:
            embeddings = model.encode(
                sentences,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                device=device,
            )
        except RuntimeError:
            batch_size //= 2
            del embeddings
            torch.cuda.empty_cache()
            gc.collect()
            print(f"Batch size reduced to {batch_size}")
            continue
        break
    return embeddings


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    optional = parser._action_groups.pop()
    required = parser.add_argument_group("required arguments")
    required.add_argument(
        "-i",
        "--input",
        help="Input sentences as lines of plaintext file",
        required=True,
    )
    required.add_argument("-o", "--output", help="Output pt path", required=True)
    required.add_argument(
        "-s", "--source_lang", help="Source language identifier (eg. ta)", required=True
    )
    args = parser.parse_args()

    F_PATH = args.input
    OUT_PATH = args.output
    SOURCE_LANG = args.source_lang
    TOTAL_SIZE = count_lines(F_PATH)
    CHUNK_SIZE = 51200

    model = SentenceTransformer("sentence-transformers/LaBSE")

    hk_embeddings, sentences, count = None, [], 0
    dataset_count = processed_count = skip_count = 0
    print(f"{TOTAL_SIZE} lines found")
    print("Starting embedding creation.")
    gen = get_line(F_PATH)
    is_complete = False

    s_last_line = last_line = None
    tm = time()

    while True:
        CHUNK_SIZE = (
            CHUNK_SIZE
            if (TOTAL_SIZE - processed_count) > CHUNK_SIZE
            else TOTAL_SIZE - processed_count
        )
        sentences = [" ".join(next(gen).split()) for _ in range(CHUNK_SIZE)]
        if processed_count + CHUNK_SIZE == TOTAL_SIZE:
            print("Completed Reading All sentences from file.")
            is_complete = True

        embeddings = compute_embeddings(
            model=model, sentences=sentences, batch_size=6144, device=device
        )

        hk_embeddings = (
            embeddings
            if hk_embeddings is None
            else np.concatenate((hk_embeddings, embeddings), axis=0)
        )
        processed_count += len(sentences)

        print(f"Embeddings shape: {hk_embeddings.shape}")
        print(
            f"Processed {((processed_count / TOTAL_SIZE) * 100)}% sentences successfully."
        )
        print(f"Time taken to add {CHUNK_SIZE} sentences is : {(time() - tm):.3f} secs")

        if is_complete:
            break

        del embeddings
        torch.cuda.empty_cache()

    if hk_embeddings is not None:
        print(f"Saving embeddings to file {OUT_PATH}")
        np.save(OUT_PATH, hk_embeddings, allow_pickle=True)

    print(f"{processed_count} Embeddings computed and Stored successfully.")
