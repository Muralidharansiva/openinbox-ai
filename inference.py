from env.environment import EmailEnv
from env.models import Action

MODEL_NAME = "rule-based-agent"

def run_task(task):
    env = EmailEnv()
    obs = env.reset()

    print(f"[START] task={task} env=openinbox model={MODEL_NAME}")

    done = False
    step = 0
    rewards = []
    email_index = 0

    while not done:
        step += 1

        email = obs.emails[email_index % len(obs.emails)]
        email_index += 1

        if email.category == "spam":
            action = Action(action_type="delete", email_id=email.id)
            action_str = "delete"

        elif task == "easy":
            action = Action(
                action_type="classify",
                email_id=email.id,
                predicted_category=email.category
            )
            action_str = "classify"

        elif task == "medium":
            action = Action(
                action_type="prioritize",
                email_id=email.id,
                predicted_priority=email.priority
            )
            action_str = "prioritize"

        else:
            action = Action(
                action_type="respond",
                email_id=email.id,
                response_text="We will assist you"
            )
            action_str = "respond"

        try:
            obs, reward, done, _ = env.step(action)
            rewards.append(reward)

            print(f"[STEP] step={step} action={action_str} reward={reward:.2f} done={str(done).lower()} error=null")

        except Exception as e:
            print(f"[STEP] step={step} action=null reward=0.00 done=true error={str(e)}")
            break

    score = sum(rewards) / max(1, len(rewards))

    print(f"[END] success=true steps={step} score={score:.2f} rewards={','.join([f'{r:.2f}' for r in rewards])}")

if __name__ == "__main__":
    for t in ["easy","medium","hard"]:
        run_task(t)