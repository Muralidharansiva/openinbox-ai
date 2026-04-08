from typing import List
from env.models import Email, Observation, Action, StepResult

class EmailEnv:
    def __init__(self, max_steps: int = 10):
        self.max_steps = max_steps
        self._emails: List[Email] = []
        self._step = 0
        self._done = False
        self._score = 0.0

    def reset(self) -> Observation:
        self._step = 0
        self._done = False
        self._score = 0.0
        self._emails = self._generate_emails()
        return self.state()

    def state(self) -> Observation:
        return Observation(emails=self._emails, step_count=self._step)

    def step(self, action: Action) -> StepResult:
        if self._done:
            return StepResult(observation=self.state(), reward=0.0, done=True, info={})

        self._step += 1
        reward = 0.0

        target = next((e for e in self._emails if e.id == action.email_id), None)

        if action.action_type == "classify" and target:
            reward += 0.3 if action.predicted_category == target.category else -0.1

        elif action.action_type == "prioritize" and target:
            reward += 0.25 if action.predicted_priority == target.priority else -0.1

        elif action.action_type == "respond" and target:
            if target.category != "spam" and action.response_text:
                reward += 0.3
                target.handled = True
            else:
                reward -= 0.2

        elif action.action_type == "delete" and target:
            if target.category == "spam":
                reward += 0.3
                target.handled = True
            else:
                reward -= 0.2

        if self._step >= self.max_steps:
            self._done = True

        self._score += reward

        return StepResult(
            observation=self.state(),
            reward=reward,
            done=self._done,
            info={"total": self._score}
        )

    def _generate_emails(self) -> List[Email]:
        return [
            Email(id="1", subject="Buy now", body="Cheap deal", priority="low", category="spam"),
            Email(id="2", subject="Meeting", body="Team sync", priority="high", category="work"),
            Email(id="3", subject="Login issue", body="Help needed", priority="medium", category="support"),
        ]