#!/bin/sh
sudo docker container stop public-client-70b
sudo docker container remove public-client-70b
sudo docker create -p 8181:8181 --ipc host --gpus 1 --volume petals-cache3:/root/.cache -e INITIAL_PEERS='/ip4/51.79.102.103/tcp/31337/p2p/QmT3TtHZyKGHuXzgWaC5AXscQsFRrH9jJGU8PC4YJUwD5g' --name public-client-70b public-chat-70b:latest
sudo docker container start public-client-70b