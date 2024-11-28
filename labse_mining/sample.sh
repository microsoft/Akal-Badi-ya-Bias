#!/bin/bash

# IndicCorp v2 File with sentences
INPUT_FILE="hi.txt"

# Output file for the sampled sentences
OUTPUT_FILE="hi_sampled.txt"

# Number of sentences to sample
NUM_SENTENCES=10000000

# Sample the sentences
shuf -n $NUM_SENTENCES "$INPUT_FILE" > "$OUTPUT_FILE"
