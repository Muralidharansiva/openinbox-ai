from env.models import Email, Observation, Action

class EmailEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        self.step_count = 0
        self.done = False
        self.score = 0
        self.emails = self._data()
        return self.state()

    def state(self):
        return Observation(emails=self.emails, step=self.step_count)

    def step(self, action: Action):
        self.step_count += 1
        reward = 0

        for e in self.emails:
            if e.id == action.email_id:

                if action.action_type == "classify":
                    reward += 0.3 if action.predicted_category == e.category else -0.1

                if action.action_type == "prioritize":
                    reward += 0.25 if action.predicted_priority == e.priority else -0.1

                if action.action_type == "delete":
                    reward += 0.3 if e.category == "spam" else -0.2
                    e.handled = True

                if action.action_type == "respond":
                    reward += 0.3 if e.category != "spam" else -0.2
                    e.handled = True

        self.score += reward

        # better done condition
        if all(e.handled or e.category == "spam" for e in self.emails):
            self.done = True

        if self.step_count >= 10:
            self.done = True

        return self.state(), reward, self.done, {"score": self.score}

    def _data(self):
        return [
            Email(id="1", subject="Offer", body="Buy now", priority="low", category="spam"),
            Email(id="2", subject="Meeting", body="Tomorrow", priority="high", category="work"),
            Email(id="3", subject="Help", body="Login", priority="medium", category="support"),
        ]