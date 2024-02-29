#!/bin/bash

sudo docker stop dell-client-65b
sudo docker rm -v dell-client-65b
sudo docker image rm hive-chat-dell-65b
cp /home/ubuntu/hiveDisk/hive-disk-python/hivedisk_api.py ../chat.petals.dev/
sudo docker build . -f Dockerfile.DELL -t hive-chat-dell-65b:latest
sh start-dell-client.sh 
