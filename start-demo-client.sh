#!/bin/sh
sudo docker container stop demo-client
sudo docker container remove demo-client
sudo docker create -p 8989:8989 --ipc host -e INITIAL_PEERS='/ip4/57.128.107.69/tcp/31637/p2p/QmYb32DhTE5hUtxcTPwUhecouTb6zrbdkGDBt5W4b3Lkvw' --volume petals-cache3:/root/.cache --volume source:/home/chat.petals.dev/SOURCE_DOCUMENTS --gpus 1 --name demo-client demo-chat:latest
sudo docker container start demo-client