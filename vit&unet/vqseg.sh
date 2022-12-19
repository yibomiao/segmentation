#!/usr/bin/env bash

if [ "$#" -eq  "0" ]
   then
     python3 vqseg12_13.py \
     --p_config "configs/vqseg.yaml"
 else
     python3 vqseg12_13.py \
     --p_config "configs/vqseg.yaml" \
     --p_state_dict "$1"
fi
