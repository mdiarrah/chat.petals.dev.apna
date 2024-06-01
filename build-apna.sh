sudo docker stop demo-client
sudo docker rm -v demo-client
sudo sudo docker build . -f Dokerfile.APNA -t demo-chat:latest
sh start-demo-client.sh 
