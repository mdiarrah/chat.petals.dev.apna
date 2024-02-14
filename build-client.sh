#!/bin/bash

sudo docker stop hive-client-llama7b
sudo docker rm -v hive-client-llama7b
sudo docker image rm hive-chat-hive-llama7b
cp /home/ubuntu/hiveDisk/hive-disk-python-techno/hivedisk_api.py chat.petals.dev/
sudo docker build . -f Dockerfile.HIVE -t hive-chat-hive-llama7b:latest
sh start-client.sh 
