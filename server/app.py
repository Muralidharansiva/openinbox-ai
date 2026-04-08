from fastapi import FastAPI
from env.environment import EmailEnv
from env.models import Action

def create_app():
    app = FastAPI()
    env = EmailEnv()

    @app.get("/")
    def root():
        return {"status": "ok"}

    @app.post("/reset")
    def reset():
        return env.reset()

    @app.post("/step")
    def step(action: Action):
        return env.step(action)

    @app.get("/state")
    def state():
        return env.state()

    return app

app = create_app()

def main():
    return app