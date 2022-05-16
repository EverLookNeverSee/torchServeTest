#!/bin/bash

sudo docker run --rm -it \
-p 3000:8080 -p 3001:8081 \
-v $(pwd)/model-store:/home/model_server/model-store pytorch/torchserve:latest \
torchserve --start --model-store model-store --models resnet34=resnet34.mar
