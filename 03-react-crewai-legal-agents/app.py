from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from agents_team import execute_agents_team  

app = FastAPI()

class TopicRequest(BaseModel):
    topic: str

@app.post("/execute")
def dsa_executa_agentes(request: TopicRequest):
    resultado = execute_agents_team(request.topic)
    return {"result": resultado}

# Local execution
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
