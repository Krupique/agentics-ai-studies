docker build -t crewai-multiagent .

docker run -d --name crewai-multiagent-deploy -p 8000:8000 --env-file .env crewai-multiagent

python client.py