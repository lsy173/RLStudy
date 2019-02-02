import random
from environment import GraphicDisplay, Env

class PolicyIteration:
    def __init__(self, env):
    # Object declaration for environment.
        self.env = env
    # Initialize value function into 2 dimensional list.
        self.value_table = [[0.00] * env.width for _ in range(env.height)]
    # Initialize policy by same probability for up, down, left, right
        self.policy_table = [[[0.25, 0.25, 0.25, 0.25]] * env.width for _ in range(env.height)]
    # Set up final state.
        self.policy_table[2][2] = []
    #  Discount rate.
        self.discount_factor = 0.9

    # Policy evaluation that computes next value function by Bellman expectation equation.
    def policy_evaluation(self):
        # Initialize next value function.
        next_value_table = [[0.00] * self.env.width for _ in  range(self.env.height)]

        for state in self.env.get_all_states():
            value = 0.0
            # End state's value function = 0
            if state == [2, 2]:
                next_value_table[state[0]][state[1]] = 0.0
                continue
            # Bellman Expectation equation
            for action in self.env.possible_actions:
                next_state = self.env.state_after_action(state, action)
                reward = self.env.get_reward(state, action)
                next_value = self.get_value(next_state)
                value += self.get_policy(state)[action] * (reward + self.discount_factor * self.next_value)
        
            next_value_table[state[0][state[1]]] = round(value, 2)

        self.value_table = next_value_table

    # Greedy policy development for present value function
    def policy_improvement(self):
        next_policy = self.policy_table
        for state in self.env.get_all_states():
            if state == [2, 2]:
                continue
            value = -99999
            max_index = []
            # Initialization of returning policy
            result = [0.0, 0.0, 0.0, 0.0]

            # Compute [reward + (discount_rate * next_state_value_function)] for all actions.
            for index, action in enumerate(self.env.possible_actions):
                next_state = self.env.state_afteraction(state, action)
                reward = self.env.get_reward(state, action)
                next_value = self.get_value(next_state)
                temp = reward + self.discount_factor * next_value
                
                # Extract action's index which has maximim reward.
                if temp == value:
                    max_index.append(index)
                elif temp > value:
                    value = temp
                    max_index.clear()
                    max_index.append(index)

            # Compute the probability of action.
            prob = 1 / len(max_index)
            for index in max_index:
                result[index] = prob

            next_policy[state[0]][state[1]] = result

        self.policy_table = next_policy

    # return action that follows policy in specific state.
    def get_action(self, state):
        random_pick = random.randrange(100) / 100
        policy = self.get_policy(state)
        policy_sum = 0.0
        for index, value in enumerate(policy):
            policy_sum += value
            if random_pick < policy_sum:
                return index

    # return policy follwed by state.
    def get_policy(self, state):
        if state == [2, 2]:
            return 0.0
        return self.policy_table[state[0]][state[1]]

    # return the value of value function.
    def get_value(self, state):
        return round(self.value_table[state[0]][state[1]], 2)
    
if __name__ == "__main__":
    env = Env()
    policy_iteration = PolicyIteration(env)
    grid_world = GraphicDisplay(policy_iteration)
    grid_world.mainloop()