#!/bin/bash

sudo docker stop dell-client-70b
sudo docker rm -v dell-client-70b
sudo docker image rm hive-chat-dell-70b
cp /home/ubuntu/hiveDisk/hive-disk-python/hivedisk_api.py .
sudo docker build . -f Dockerfile.DELL -t hive-chat-dell-70b:latest
sh start-dell-client.sh 
