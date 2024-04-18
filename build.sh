#!/bin/bash

sudo docker stop public-client-70b
sudo docker rm -v public-client-70b
sudo docker image rm public-chat-70b
sudo docker build . -f Dockerfile -t public-chat-70b:latest
sh start-client.sh 