#!/bin/bash

sudo docker stop public-client
sudo docker rm -v public-client
sudo docker image rm public-chat
sudo docker build . -f Dockerfile -t public-chat:latest
sh start-client.sh 