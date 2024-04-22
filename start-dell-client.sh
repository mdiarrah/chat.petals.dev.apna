#!/bin/sh
sudo docker container stop dell-client
sudo docker container remove dell-client
sudo docker create -p 8989:8989 --ipc host -e INITIAL_PEERS='/ip4/51.79.102.103/tcp/39337/p2p/Qma7LNTwU6MtR7CqetNuJbqWBnQ9YYQEjM6v9vQfG3bAhC' --gpus 1 --volume petals-cache3:/root/.cache --name dell-client hive-chat-dell:latest
sudo docker container start dell-client
