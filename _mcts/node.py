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
        self.layer = 0
        self.parent = parent
        self.N = 0  # times this position was visited
        self.V = 0
        self.children = dict()

    def initialize_child(self, action):
        if action not in self.children:
            self.children[action] = None
            return True
        return False

    def add_child(self, action):
        if action not in self.children or not self.children[action]:
            child_node = Node(action)
            child_node.parent = self
            child_node.layer = self.layer + 1
            self.children[action] = child_node
            return True
        return False

    def initialize_children(self, actions: list):
        new_children = False
        for action in actions:
            new_child = self.initialize_child(action)
            new_children = new_children or new_child
        return new_children

    def add_children(self, actions: list):
        for action in actions:
            self.add_child(action)

    def prune(self, child_to_keep):
        # self.children = {child_to_keep: self.children[child_to_keep]}
        self.children = {k: v for (k, v) in self.children.items() if v is not None and v.action == child_to_keep}
        for child in self.children:
            self.children[child].V = self.children[child].V/self.children[child].N
            self.children[child].N = 1
            # self.children[child].V = 0
            # self.children[child].N = 0

    # def merge(self, primary_child, secondary_child):
    #     if primary_child in self.children and secondary_child in self.children:
    #         self.children[secondary_child] = self.children[primary_child]
    #
    # def greedy_merge(self, final_layer, margin=0, merge_to=None):
    #     if final_layer:
    #         return
    #     visited_children = {k: v for (k, v) in self.children.items() if v is not None}
    #     vs_next = np.array([child.V for child in list(visited_children.values())])
    #     ns_next = np.array([(child.N or 1) if child is not None else 1 for child in visited_children.values()])
    #     vs_next_avg = vs_next / ns_next
    #     # amax_index = np.where(vs_next_avg == np.max(vs_next_avg))[0]
    #     amax_index = np.where(np.logical_and(
    #         vs_next_avg >= np.max(vs_next_avg) * (1 - margin),
    #         vs_next_avg <= np.max(vs_next_avg) * (1 + margin)))[0]
    #
    #     if len(amax_index) > 1:
    #         primary_child_index = secure_random.choice(amax_index)
    #         amax_index = amax_index[amax_index != primary_child_index]
    #
    #         visited_children_actions = list(visited_children.keys())
    #         primary_child = self.children[visited_children_actions[primary_child_index]]
    #
    #         # print('merging ', visited_children_actions[primary_child_index])
    #         for child_index in amax_index:
    #             self.merge(visited_children_actions[primary_child_index], visited_children_actions[child_index])
    #         return primary_child
    #     return None

    def random_child(self):
        child_key = secure_random.choice(list(self.children.keys()))
        self.add_child(child_key)
        return self.children[child_key]

    # def score_ucb(self, c):
    #     return self.V/(self.N or 1) + c * np.sqrt(np.log(self.parent.N or 1) / self.N)

    def preferred_child_ucb(self, c, final_layer=False):
        # UCB formula: V_ucb_next = V + c*sqrt(ln(N_s)/n_s_next)
        if not final_layer:
            vs_next = np.array([child.V if child is not None else np.inf for child in list(self.children.values())])
        else:
            vs_next = np.zeros(len(self.children))

        ns_next = np.array([(child.N or 1) if child is not None else 1 for child in self.children.values()])
        # qs = self.V + vs_next
        q_ucb = vs_next/ns_next + c * np.sqrt(np.log(self.N or 1) / ns_next)
        # making sure we pick the maximums at random
        a_ucb_index = secure_random.choice(np.where(q_ucb == np.max(q_ucb))[0])
        child_key = list(self.children.keys())[a_ucb_index]
        self.add_child(child_key)

        # qs_next = np.array([child.score_ucb(c) if child is not None else np.inf for child in self.children.values()])
        # a_ucb_index = secure_random.choice(np.where(qs_next == np.max(qs_next))[0])
        # child_key = list(self.children.keys())[a_ucb_index]
        # self.add_child(child_key)
        return self.children[child_key]

    def preferred_child_greedy(self, final_layer=False, prefer="value"):
        # {key: value for (key, value) in dictOfNames.items() if key % 2 == 0}
        visited_children = {k: v for (k, v) in self.children.items() if v is not None}
        # print(visited_children)
        if not visited_children:
            child_key = secure_random.choice(list(self.children.keys()))
            self.add_child(child_key)
            return self.children[child_key]

        if not final_layer:
            vs_next = np.array([child.V for child in list(visited_children.values())])
            # ns_next = np.array([(child.N or 1) for child in visited_children.values()])
        else:
            vs_next = np.zeros(len(visited_children))
            # ns_next = np.ones(len(visited_children))

        # if prefer.lower() == 'value':
            #making sure we pick the maximums at random
            # vs_next_avg = vs_next / ns_next
            # amax_index = secure_random.choice(np.where(vs_next_avg == np.max(vs_next_avg))[0])
        amax_index = secure_random.choice(np.where(vs_next == np.max(vs_next))[0])
        # elif prefer.lower() == 'visits':
        #     amax_index = secure_random.choice(np.where(ns_next == np.max(ns_next))[0])

        child_key = list(visited_children.keys())[amax_index]
        return self.children[child_key]

    def backup_update(self, r, alpha=1):
        self.V += alpha * r
        self.N += 1
        if self.parent:
            self.parent.backup_update(r, alpha)
            # self.parent.backup_update(self.V)

    def backup_update_jinjerry(self, r):
        self.V = max(self.V, r)
        self.N += 1
        if self.parent:
            self.parent.backup_update(self.V)

    def reset(self):
        self.N = 0
        self.V = 0
        self.children = dict()

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