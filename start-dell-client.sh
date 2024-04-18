#!/bin/sh
sudo docker container stop dell-client-70b-stable
sudo docker container remove dell-client-70b-stable
sudo docker create -p 9191:9191 --ipc host --gpus 1 --volume petals-cache3:/root/.cache -e INITIAL_PEERS='/ip4/51.79.102.103/tcp/31337/p2p/QmT3TtHZyKGHuXzgWaC5AXscQsFRrH9jJGU8PC4YJUwD5g' --name dell-client-70b hive-chat-dell-70b-stable:latest
sudo docker container start dell-client-70b-stable
