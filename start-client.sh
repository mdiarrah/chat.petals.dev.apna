#!/bin/sh
sudo docker container stop hive-client-llama7b
sudo docker container remove hive-client-llama7b
sudo docker create -p 8282:8282 --ipc host -e INITIAL_PEERS='/ip4/51.79.102.103/tcp/39337/p2p/Qma7LNTwU6MtR7CqetNuJbqWBnQ9YYQEjM6v9vQfG3bAhC' --gpus 1 --volume petals-cache3:/root/.cache --name hive-client-llama7b hive-chat-hive-llama7b:latest
sudo docker container start hive-client-llama7b
