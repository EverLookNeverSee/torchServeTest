#!/bin/bash

torch-model-archiver --model-name resnet34 \
--version 0.0.1 \
--serialized-file resnet34.pt \
--extra-files ./index_to_name.json,./handler.py \
--handler handler.py  \
--export-path model_store -f
