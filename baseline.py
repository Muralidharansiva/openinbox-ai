from env.environment import EmailEnv
from env.models import Action

def normalize(x):
    return max(0.0, min(1.0, x/3))

def run_easy():
    env = EmailEnv()
    obs = env.reset()
    total = 0
    for e in obs.emails:
        total += env.step(Action(action_type="classify", email_id=e.id, predicted_category=e.category))[1]
    return normalize(total)

def run_medium():
    env = EmailEnv()
    obs = env.reset()
    total = 0
    for e in obs.emails:
        total += env.step(Action(action_type="prioritize", email_id=e.id, predicted_priority=e.priority))[1]
    return normalize(total)

def run_hard():
    env = EmailEnv()
    obs = env.reset()
    total = 0
    for e in obs.emails:
        total += env.step(Action(action_type="classify", email_id=e.id, predicted_category=e.category))[1]
        if e.category == "spam":
            total += env.step(Action(action_type="delete", email_id=e.id))[1]
        else:
            total += env.step(Action(action_type="respond", email_id=e.id, response_text="ok"))[1]
    return normalize(total)

def run_baseline():
    return {
        "easy": run_easy(),
        "medium": run_medium(),
        "hard": run_hard()
    }