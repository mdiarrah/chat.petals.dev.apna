#!/bin/bash

sudo docker stop dell-client-70b-stable
sudo docker rm -v dell-client-70b-stable
sudo docker image rm hive-chat-dell-70b-stable
cp /home/ubuntu/hiveDisk/hive-disk-python/hivedisk_api.py .
sudo docker build . -f Dockerfile.DELL -t hive-chat-dell-70b-stable:latest
sh start-dell-client.sh 
