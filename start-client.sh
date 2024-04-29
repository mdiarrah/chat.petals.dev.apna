#!/bin/sh
sudo docker container stop public-client
sudo docker container remove public-client
sudo docker create -p 8181:8181 --ipc host --gpus 1 --volume petals-cache3:/root/.cache -e INITIAL_PEERS='/ip4/51.79.102.103/tcp/39337/p2p/Qma7LNTwU6MtR7CqetNuJbqWBnQ9YYQEjM6v9vQfG3bAhC' --name public-client public-chat:latest
sudo docker container start public-client