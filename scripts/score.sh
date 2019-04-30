#!/usr/bin/env bash

EMB=/home/cgsdfc/embeddings/word2vec/GoogleNews_negative300/GoogleNews-vectors-negative300.bin
PRED=/home/cgsdfc/UbuntuDialogueCorpus/ResponseContextPairs/ModelPredictions/VHRED/First_VHRED_BeamSearch_5_GeneratedTestResponses.txt_First.txt
GT=/home/cgsdfc/UbuntuDialogueCorpus/ResponseContextPairs/raw_testing_responses.txt
PREFIX=/home/cgsdfc/Result/Test
CONFIG="-A -X -G"

embedding_metrics.py
    -embeddings $EMB \
    -predicted $PRED \
    -ground_truth $GT \
    -p $PREFIX

