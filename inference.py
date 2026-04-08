from env.environment import EmailEnv
from env.models import Action

def run():
    env = EmailEnv()
    obs = env.reset()

    for e in obs.emails:
        if e.category == "spam":
            env.step(Action(action_type="delete", email_id=e.id))
        else:
            env.step(Action(action_type="respond", email_id=e.id, response_text="Handled"))

    print("Inference completed")

if __name__ == "__main__":
    run()