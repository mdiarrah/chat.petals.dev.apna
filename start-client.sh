#!/bin/sh
sudo docker container stop hive-client-llama7b
sudo docker container remove hive-client-llama7b
sudo docker create -p 9090:9090 --ipc host --gpus 1 --volume petals-cache3:/root/.cache --name hive-client-llama7b hive-chat-hive-llama7b:latest
sudo docker container start hive-client-llama7b
