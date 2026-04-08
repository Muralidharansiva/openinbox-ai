from fastapi import FastAPI
from env.environment import EmailEnv
from env.models import Action

def create_app():
    app = FastAPI()
    env = EmailEnv()

    @app.get("/")
    def root():
        return {"status": "running"}

    @app.post("/reset")
    def reset():
        return env.reset().model_dump()

    @app.post("/step")
    def step(action: Action):
        return env.step(action).model_dump()

    @app.get("/state")
    def state():
        return env.state().model_dump()

    return app

app = create_app()

def main():
    return app