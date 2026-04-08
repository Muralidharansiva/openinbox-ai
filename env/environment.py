from env.models import Email, Observation, Action, StepResult

class EmailEnv:
    def __init__(self):
        self._step = 0
        self._done = False
        self._emails = []

    def reset(self):
        self._step = 0
        self._done = False
        self._emails = self._generate_emails()
        return self.state()

    def state(self):
        return Observation(
            emails=self._emails,
            step_count=self._step
        ).model_dump()

    def step(self, action: Action):
        self._step += 1
        reward = 0.3

        return StepResult(
            observation=Observation(
                emails=self._emails,
                step_count=self._step
            ),
            reward=reward,
            done=self._step >= 10,
            info={}
        ).model_dump()

    def _generate_emails(self):
        return [
            Email(id="1", subject="Buy now", body="Cheap", priority="low", category="spam"),
            Email(id="2", subject="Meeting", body="Discuss", priority="high", category="work"),
        ]