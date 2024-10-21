class OnlineMemory:
    def __init__(self, env):
        self.batch = []  
        self.env = env

    def process_transitions(self, req):

        transitions = req["transitions"]

        '''Process transitions'''
        for transition in transitions:
            """Prepare transition"""
            raw_state = transition["state"][1:]
            raw_next_state = transition["next_state"][1:]
            raw_action = transition["action"]

            state = self.env.convert_state(raw_state)
            next_state = self.env.convert_state(raw_next_state)
            action = self.env.get_action(raw_action)
            done = transition["done"]
            reward = self.env.reward_func(state, next_state)

            transition = {
                "state": state,
                "action": action,
                "next_state": next_state,
                "reward": reward,
                "done": done,
            }

            self.batch.append(transition)

    def reset(self):
        self.batch = [] 