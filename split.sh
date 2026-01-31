#!/bin/bash
python3 generate_advanced_data.py
COUNT=`wc -l corpus_clean.txt | cut -d' ' -f1`
TAIL=$((COUNT / 10))
HEAD=$((COUNT-TAIL))
head -n $HEAD corpus_clean.txt > corpus.txt
tail -n $TAIL corpus_clean.txt > valid.txt

