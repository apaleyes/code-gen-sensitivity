class BaseModel():
    def __init__(self, name, temperature=0.0):
        self.name = name
        self.temperature = temperature

    def __call__(self, prompt):
        raise NotImplementedError()

    @property
    def call_timeout(self):
        raise NotImplementedError()
