#!/bin/bash

sudo docker stop dell-client
sudo docker rm -v dell-client
sudo docker image rm hive-chat-dell
cp /home/ubuntu/hiveDisk/hive-disk-python/hivedisk_api.py chat.petals.dev/
sudo docker build . -f Dockerfile.DELL -t hive-chat-dell:latest
sh start-dell-client.sh 
