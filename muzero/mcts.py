import math
import random
import numpy as np

# https://en.wikipedia.org/wiki/Monte_Carlo_tree_search
# Monte Carlo Tree Search
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

class Node(object):
    def __init__(self, prior: float):
        self.visit_count = 0
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.hidden_state = None
        self.reward = 0
        self.to_play = -1

    def expanded(self) -> bool:
        return len(self.children) > 0
    
    def value(self) -> float:
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

pb_c_base = 19652
pb_c_init = 1.25

discount = 0.95
root_dirichlet_alpha = 0.25
root_exploration_fraction = 0.25

class MinMaxStats(object):
    '''A class that holds the min-max values of the trees.'''

    def __init__(self):
        self.maximum = -float('inf')
        self.minimum = float('inf')

    def update(self, value: float):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)
    
    def normalize(self, value: float) -> float:
        if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value
    
    # This score for a node is based on its value, plus an exploration bonus based on the prior
    # https://stackoverflow.com/questions/57359229/ucb-formula-for-monte-carlo-tree-search-when-score-is-between-0-and-n
    def ucb_score(parent: Node, child: Node, min_max_stats=None) -> float:
        pb_c = math.log((parent.visit_count + pb_c_base + 1) / pb_c_base) + pb_c_init
        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

        prior_score = 0
        if child.visit_count > 0:
            if min_max_stats is not None:
                value_score = child.reward + discount * min_max_stats.normalize(child.value())
            else:
                value_score = child.reward + discount * child.value()
        else:
            value_score = 0
        
        return prior_score + value_score
    
    def select_child(node: Node, min_max_stats: None):
        out = [(ucb_score(node, child, min_max_stats), action, child)
                for action, child in node.children.items()]
        smax = max([x[0] for x in out])
        # this max is why it favors 1's over 0's
        _, action, child = random.choice(list(filter(lambda x: x[0] == smax, out)))
        return action, child

    def mcts_search(m, observation, min_simulations=10, minmax=True):
        # init the root node
        root = Node(0)
        root.hidden_state = m.ht(observation)
        