class State:
    class Agent:
        def __init__(self):
            self.memory: str = "无相关记忆"
            self.is_silent: bool = True
            self.unread_events: list = []

    class Env:
        def __init__(self):
            self.is_speaking: bool = False
            self.input: dict = {
                "content": "",
                "source": "",
                "ephemeral_context": False,
                "timestamp": None
            }

    def __init__(self):
        self.agent = State.Agent()
        self.env = State.Env()
