class BaseModel():
    def __init__(self, name):
        self.name = name

    def get_code(self, prompt):
        raise NotImplementedError()

    @property
    def call_timeout(self):
        raise NotImplementedError()