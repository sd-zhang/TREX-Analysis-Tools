import numpy as np
from _utils.utils import secure_random
class Node:
    """
    Node for the MCTS. Stores the move applied to reach this node from its parent,
    stats for the associated game position, children, parent and outcome
    (outcome==none unless the position ends the game).
    Args:
        move:
        parent:
        N (int): times this position was visited.
        Q (int): average reward (wins-losses) from this position.
        children (dict): list of successive nodes.
    """

    def __init__(self, action: int = None, parent: object = None):
        """
        Initialize a new node with optional move and parent and initially empty
        children list and rollout statistics and unspecified outcome.

        """
        self.action = action
        self.parent = parent
        self.N = 0  # times this position was visited
        self.V = 0
        self.children = dict()

    def initialize_child(self, action):
        if action not in self.children:
            self.children[action] = None

    def add_child(self, action):
        if action not in self.children or not self.children[action]:
            child_node = Node(action)
            child_node.parent = self
            self.children[action] = child_node

    def initialize_children(self, actions: list):
        for action in actions:
            self.initialize_child(action)

    def add_children(self, actions: list):
        for action in actions:
            self.add_child(action)

    def random_child(self):
        child_key = secure_random.choice(list(self.children.keys()))
        self.add_child(child_key)
        return self.children[child_key]

    def preferred_child_ucb(self, c, final_layer=False):
        # UCB formula: V_ucb_next = V + c*sqrt(ln(N_s)/n_s_next)
        if not final_layer:
            vs_next = np.array([child.V if child is not None else np.inf for child in list(self.children.values())])
        else:
            vs_next = np.zeros(len(self.children))

        ns_next = np.array([(child.N or 1) if child is not None else 1 for child in self.children.values()])
        qs = self.V + vs_next
        q_ucb = qs + c * np.sqrt(np.log(self.N or 1) / ns_next)
        #making sure we pick the maximums at random
        a_ucb_index = secure_random.choice(np.where(q_ucb == np.max(q_ucb))[0])
        child_key = list(self.children.keys())[a_ucb_index]
        self.add_child(child_key)
        return self.children[child_key]

    def preferred_child_greedy(self, final_layer=False):
        # {key: value for (key, value) in dictOfNames.items() if key % 2 == 0}
        visited_children = {k: v for (k, v) in self.children.items() if v is not None}
        if not final_layer:
            vs_next = np.array([child.V for child in list(visited_children.values())])
        else:
            vs_next = np.zeros(len(visited_children))

        #making sure we pick the maximums at random
        amax_index = secure_random.choice(np.where(vs_next == np.max(vs_next))[0])
        child_key = list(visited_children.keys())[amax_index]
        return self.children[child_key]

    def backup_update(self, r):
        self.V += r
        self.N += 1
        if self.parent:
            self.parent.backup_update(r)

    def backup_update_jinjerry(self, r):
        self.V = max(self.V, r)
        self.N += 1
        if self.parent:
            self.parent.backup_update(self.V)

    #
    # def add_child(self, child):

        #
        # if self.N == 0:
        #     return 0 if explore == 0 else GameMeta.INF
        # else:
        #     return self.Q / self.N + explore * np.sqrt(2 * np.log(self.parent.N) / self.N)  # exploitation + exploration

# import numpy as np
# from _mcts import node
# game_tree = dict()
# game_tree['root_node'] = node.Node()
# game_tree['current_node'] = game_tree['root_node']
# actions = list(range(3))
# game_tree['current_node'].add_children(actions)
# # next_node = game_tree['current_node'].random_child()
# next_node = game_tree['current_node'].preferred_child_ucb(1)