from fastapi import FastAPI
from env.environment import EmailEnv
from env.models import Action
from baseline import run_baseline

app = FastAPI()

env = EmailEnv()
last_score = 0

@app.post("/reset")
def reset():
    return env.reset()

@app.post("/step")
def step(action: Action):
    global last_score
    obs, reward, done, info = env.step(action)
    last_score = info.get("score", 0)
    return {
        "observation": obs,
        "reward": reward,
        "done": done,
        "info": info
    }

@app.get("/state")
def state():
    return env.state()

@app.get("/tasks")
def tasks():
    return {"tasks": ["easy","medium","hard"]}

@app.get("/baseline")
def baseline():
    return run_baseline()

@app.get("/grader")
def grader():
    return {"score": last_score}