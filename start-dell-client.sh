#!/bin/sh
sudo docker container stop dell-client-70b
sudo docker container remove dell-client-70b
sudo docker create -p 9090:9090 --ipc host --gpus 1 --volume petals-cache3:/root/.cache -e INITIAL_PEERS='/ip4/51.79.102.103/tcp/31337/p2p/QmT3TtHZyKGHuXzgWaC5AXscQsFRrH9jJGU8PC4YJUwD5g' --name dell-client-70b hive-chat-dell-70b:latest
sudo docker container start dell-client-70b
