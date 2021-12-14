import numpy as np
import copy
from _utils.utils import secure_random

class MCTS:
    def __init__(self,
                 participants: dict,
                 learner_id: str,
                 market,
                 reward,
                 time_start: int,
                 time_end: int,
                 max_iterations: int,
                 c_adjustment):

        self.learner_id = learner_id
        self.participants = dict()
        self.learner = dict()
        self.market = market
        self.reward = reward
        self.time_start = time_start
        self.time_end = time_end
        self.max_iterations = max_iterations
        self.c_adjustment = c_adjustment
        self.game_tree = dict()

        self.update_participants(participants)

        actions = self.learner['trader']['actions']
        self.action_spaces = dict()
        self.shape_action_space = list()

        for action in actions:
            self.action_spaces[action] = len(actions[action])
            self.shape_action_space.append(len(actions[action]))

        num_individual_entries = 1
        for dimension in self.shape_action_space:
            num_individual_entries = num_individual_entries * dimension
        self.linear_action_space = np.arange(num_individual_entries).tolist()

    def update_participants(self, participants):
        # self.participants = json.loads(json.dumps(participants))
        self.participants = copy.deepcopy(participants)
        self.learner.update(self.participants[self.learner_id])

    def init_game_tree(self, time_start):
        # We need a data structure to store the 'game tree'
        # A dict with a hashed list l as key; l = [ts, a1, ...an]
        # entries in the dicts are:
        # n: number of visits
        # V: current estimated value of state
        # a: a dict with action tuples as keys
        # each of those is a subdict with key
        # r: reward for transition from s -a-> s'
        # s_next: next  state
        # n: number of times this action was taken

        self.game_tree = dict()
        s_0 = self.encode_states(time=time_start)
        self.game_tree[s_0] = {'N': 0}
        # return game_tree

    def encode_states(self, time):
        # for now we only encode time
        if 'battery' in self.learner['trader']['actions']:
            if time <= self.time_start:
                state_of_charge = 0
            else:
                state_of_charge = self.learner['metrics'][time]['battery']['battery_SoC']
        else:
            state_of_charge = None
        t_next = time + 60
        s_next = (t_next, state_of_charge)
        return s_next

    # decode states, placeholder function for more complex states
    def decode_states(self, s):
        timestamp = s[0]
        return timestamp, None

    def decode_actions(self, a, timestamp):
        # actions = self.simulation_env.participants[participant]['trader']['actions']
        # action_types = [action for action in self.simulation_env.participants[participant]['metrics'][ts]]
        actions_dict = {}
        actions = self.learner['trader']['actions']
        action_types = [action for action in self.learner['metrics'][timestamp]]
        a = np.unravel_index(int(a), self.shape_action_space)
        # print(price)
        # print(a)
        for action_type in action_types:
            if action_type in {'bids', 'asks'} and actions['quantity'][a[1]]:
                actions_dict[action_types[0]] = {
                    str((timestamp-60, timestamp)): {
                        'quantity': actions['quantity'][a[1]],
                        'price': actions['price'][a[0]],
                        'source': 'solar',
                        'participant_id': self.learner_id
                        }
                    }
            elif action_type == 'battery':
                # print(actions)
                actions_dict['battery'] = {
                    'target_flux': actions['battery'][a[-1]],
                    'battery_SoC': None
                }

        return actions_dict

    def ucb(self, s_now):
        # print(c)
        # UCB formula: V_ucb_next = V + c*sqrt(ln(N_s)/n_s_next)
        N_s = self.game_tree[s_now]['N']
        c = self.c_adjustment
        all_s_next = []

        num_actions = len(self.linear_action_space)
        Q_ucb = [None] * num_actions  # determine the value of all followup transitions states

        for idx_a in range(num_actions):
            a = self.linear_action_space[idx_a]

            if a not in self.game_tree[s_now]['a']:
                n_next = 0 #if the aqction transition isnt logged, we havent sampled it yet
                Q_ucb[idx_a] = np.inf
            else:
                s_next = self.game_tree[s_now]['a'][a]['s_next']
                n_next = self.game_tree[s_now]['a'][a]['n']
                r = self.game_tree[s_now]['a'][a]['r']
                ts, _ = self.decode_states(s_next)
                if ts >= self.time_end:
                    v_next = 0
                else:
                    v_next = self.game_tree[s_next]['V']
                q = r + v_next
                Q_ucb[idx_a] = q + c * np.sqrt(np.log(N_s)/n_next)

        #making sure we pick the maximums at random
        a_ucb_index = secure_random.choice(np.where(Q_ucb == np.max(Q_ucb))[0])
        a_ucb = self.linear_action_space[a_ucb_index]
        return a_ucb

    # a single step of MCTS, one node evaluation
    def step(self, s_now):
        # see if wee arein a leaf node
        finished = False
        # check of leaf node, if leaf node then do rollout, estimate V of node
        if 'a' not in self.game_tree[s_now]:
            self.game_tree[s_now]['V'] = self.default_rollout(s_now=s_now)
            self.game_tree[s_now]['a'] = dict()
            self.game_tree[s_now]['N'] += 0

            finished = True
            s_next = None
            a = None
            return s_next, a, finished
        # its no leaf node, so we expand using ucb policy
        # determining ucb next action
        a = self.ucb(s_now=s_now)
        r, s_next = self.evaluate_transition(s_now=s_now, a=a)
        # this means we're taking a leaf node
        if a not in self.game_tree[s_now]['a']:  # equivalent to game_tree[s_now]['a'][a]['n'] == 0
            self.game_tree[s_now]['a'][a] = {'r': r,
                                             'n': 0,
                                             's_next': s_next}  # gotta mak sure all those get populated

        if self.game_tree[s_now]['a'][a]['s_next'] != s_next:
            print('encountered non causal state transitions, unforseen and might break code behavior!!!')
        # if self.game_tree[s_now]['a'][a]['r'] != r:
        #     print('encountered non cuasal reward, unforseen and might break code behavior!!')

        ts, _ = self.decode_states(s_next)
        self.game_tree[s_now]['a'][a]['n'] += 1
        self.game_tree[s_now]['N'] += 1

        # update V estimate for node
        if s_next not in self.game_tree and ts <= self.time_end:
            self.game_tree[s_next] = {'N': 0}

        if ts > self.time_end:
            finished = True

        return s_next, a, finished

    # one MCTS rollout
    def one_rollout_and_backup(self, s_now):
        action = None
        trajectory = []
        finished = False
        s_now = s_now

        # we're traversing the tree till we hit bottom
        #ToDo: change pieces of code so we can actually do this with a persistent tree?
        while not finished:
            trajectory.append((s_now, action))
            s_now, action, finished = self.step(s_now=s_now)
        self.bootstrap_values(trajectory)

    # backprop of values
    def bootstrap_values(self, trajectory):
        # now we backpropagate the value up the tree:
        if len(trajectory) > 1:
            trajectory.reverse()

        for idx in range(len(trajectory)):
            s_now = trajectory[idx][0]
            # get all possible followup states
            Q = []

            for a in self.game_tree[s_now]['a']:
                r = self.game_tree[s_now]['a'][a]['r']
                s_next = self.game_tree[s_now]['a'][a]['s_next']

                if s_next in self.game_tree:
                    V_s_next = self.game_tree[s_next]['V']
                else:
                    V_s_next = 0
                Q.append(r + V_s_next)
            if Q != []:
                self.game_tree[s_now]['V'] = np.amax(Q)

    def evaluate_transition(self, s_now, a):
        # for now the state tuple is: (time)
        timestamp, _ = self.decode_states(s_now)  # _ being a placeholder for now
        actions = self.decode_actions(a=a, timestamp=timestamp)
        self.learner['metrics'][timestamp].update(actions)
        r, _, _, _, _, _, _, _ = self.get_reward_for_transactions(timestamp=timestamp)
        s_next = self.encode_states(time=timestamp)
        return r, s_next

    def one_default_step(self, s_now):
        # here's the two policies that we'll be using for now:
        # UCB for the tree traversal
        # random action selection for rollouts
        finished = False
        a = secure_random.choice(self.linear_action_space)
        ts, _ = self.decode_states(s_now)
        r, s_next = self.evaluate_transition(s_now=s_now,
                                             a=a)
        if ts == self.time_end:
            finished = True
        return r, s_next, a, finished

    def default_rollout(self, s_now):
        finished = False
        v = 0
        while not finished:
            r, s_next, _, finished = self.one_default_step(s_now=s_now)
            s_now = s_next
            v += r
        return v

# get the market settlement for one specific row for one specific agent from self.simulation_env.participants
    def get_reward_for_transactions(self, timestamp):
        # get the market ledger
        simulated_transactions = self.market.simulate_transactions(participants=self.participants,
                                                                   learner_id=self.learner_id,
                                                                   timestamp=timestamp)
        market_ledger = list()
        # quantity = 0

        for index in range(simulated_transactions.shape[0]):
            settlement = simulated_transactions.iloc[index]
            # quantity = settlement['quantity']
            entry = self.market.simulated_transactions_to_ledger(settlement, self.learner_id)
            if entry is not None:
                market_ledger.append(entry)

        # if market_ledger:
        #     print(simulated_transactions)

        # if quantity:
        #     print(quantity)
        # ToDO: test if market is actually doing the right thing

        # we need access to start_energy [0 ... max_energy] and a target_action [-max_energy, max_energy]
        real_flux = 0
        if 'battery' in self.learner['metrics'][timestamp]:
            # print(self.learner_id, self.learner['metrics'][timestamp])
            # FixMe: Apparently Daniel fucked up time here somehow, the very first row of Metrics never gets updated to a real So
            if timestamp-60 <= self.time_start:
                soc_start = 0
            else:
                if timestamp-60 not in self.learner['metrics']: # FixMe: catch for general shit
                    print('missing ts!!')
                    print(timestamp)
                soc_start = self.learner['metrics'][timestamp-60]['battery']['battery_SoC']
            target_flux = self.learner['metrics'][timestamp]['battery']['target_flux']

            # seems like this is error prone somehow?!
            if soc_start is None:  # toDo: catch and fix, once this area is debugged get rid
                print('aha, need to debug')
                soc_start = 0
            if target_flux is None:
                target_flux = 0

            real_flux, soc_end = self.learner['storage'].simulate_activity(start_energy=soc_start, target_energy=target_flux)
            self.learner['metrics'][timestamp]['battery']['battery_SoC'] = soc_end
            # print(soc_start, target_flux, real_flux, soc_end)

            # if self.learner_id == "R1" and timestamp-60 == self.time_start:
            #     print(timestamp, soc_start, target_flux, real_flux)


        # calculate the resulting grid transactions
        generation = self.learner['metrics'][timestamp]['gen']
        consumption = self.learner['metrics'][timestamp]['load']

        # if self.learner_id == "R2":
        #     print(self.learner_id, timestamp, generation, consumption, soc_start, target_flux, real_flux)
        #     print(self.learner['metrics'][timestamp])
        #     print(self.learner['metrics'][timestamp - 60])

        bids, asks, grid_transactions, financial_transactions = \
            self.market.deliver(market_ledger=market_ledger,
                                generation=generation,
                                consumption=consumption,
                                battery=real_flux)

        # if self.learner_id == 'R2':
        #     print(market_ledger, generation, consumption, real_flux)
        # print(bids, asks, grid_transactions, financial_transactions)

        # if self.learner_id == "R1":
        #     print(self.learner_id, timestamp, generation, consumption)
        #     print(soc_start, target_flux, real_flux)
        #     print(bids, asks, grid_transactions, financial_transactions)
        #     print(self.learner['metrics'][timestamp - 60]['battery'])
        #     print(timestamp)
            # print(generation, consumption)
            # print(self.learner['metrics'][timestamp - 60]['asks'][str((timestamp-60, timestamp))]['quantity'])

        # if self.learner_id == "R3":
            # print(generation, consumption)
        # #     print(self.learner_id, timestamp, generation, consumption)
        # #     print(soc_start, target_flux, real_flux)
        # #     print(bids, asks, grid_transactions, financial_transactions)
        #     print(self.learner['metrics'][timestamp - 60])
        #     print(timestamp)
        # access_fee = 0 # temporary hack to discourage market use when not necessary
        # bid_action_qty = 0
        # ask_action_qty = 0
        # if 'bids' in self.learner['metrics'][timestamp]:
        #     bid_action_qty = self.learner['metrics'][timestamp]['bids'][str((timestamp-60, timestamp))]['quantity']
        #     if consumption <= 0 and bid_action_qty >= 0:
        #         access_fee -= 10
        #
        # if 'asks' in self.learner['metrics'][timestamp]:
        #     ask_action_qty = self.learner['metrics'][timestamp]['asks'][str((timestamp-60, timestamp))]['quantity']
        #     if generation <= 0 and (ask_action_qty - real_flux) >= 0:
        #         access_fee -= 10

        # if self.learner_id == "R1":
        #     print(timestamp, generation, consumption, bid_action_qty, ask_action_qty, real_flux, access_fee)
            # print(timestamp, generation, consumption)
            # print(self.learner['metrics'][timestamp])

            # if self.learner_id == "R1":
            #     print(generation, consumption, ask_action_qty, access_fee)

        # then calculate the reward function
        rewards, metrics = self.reward.calculate(bids=bids,
                                                 asks=asks,
                                                 grid_transactions=grid_transactions,
                                                 financial_transactions=financial_transactions)

        # print(self.learner['metrics'][timestamp - 60]
        # print('r: ', rewards)
        # if do_print:
        # print('market', market_ledger)
        # print('grid', grid_transactions)
        # print('r', rewards)
        # print('metered_r', simulation_env.participants[learning_participant]['metrics']['reward'][ts])
        bids_qty = metrics.pop('bids_quantity', 0)
        asks_qty = metrics.pop('asks_quantity', 0)
        quantity = bids_qty + asks_qty

        # rewards += access_fee
        # grid_transactions = (grid_buy, self.grid_buy_price, grid_sell, self.grid_sell_price)
        # return rewards, quantity, metrics
        return rewards, metrics, bids_qty, asks_qty, grid_transactions[0], grid_transactions[2], financial_transactions[0], financial_transactions[2]

    #ToDo: make sure we reset the node visit numbers
    # make reset tree to reset, make it a string
    # default is None
    # 'tree' means we reset the whole tree
    # 'visits' means we reset visits (for tree persistence)
    # add a disclaimer if the thing
    def run(self, reset_tree=False):
        # self.init_game_tree(self.time_start)
        # if not self.game_tree:
        # print(reset_tree)
        if reset_tree:
            print('completely reset game tree')
            self.init_game_tree(self.time_start)
        else:
            print('reset visits for the game tree')
            self.reset_visits()
        s_0 = self.encode_states(time=self.time_start)
        for iteration in range(self.max_iterations):
            self.one_rollout_and_backup(s_0)

        return {self.learner_id: {
            'game_tree': self.game_tree,
            's_0': s_0,
            'metrics': self.learner['metrics']
        }}

    def reset_visits(self):
        print('.')
        for state in self.game_tree:
            # reset the visits to all s/a pairs to 1
            # reset the visits to all states to num_actions
            n_actions = 0
            for action in self.game_tree[state]['a']:
                self.game_tree[state]['a'][action]['n'] = 1
                n_actions += 1
            self.game_tree[state]['N'] = n_actions


    def greedy_policy(self, s_now):
        q = []
        actions = []
        for a in self.game_tree[s_now]['a']:
            r = self.game_tree[s_now]['a'][a]['r']
            s_next = self.game_tree[s_now]['a'][a]['s_next']
            if s_next in self.game_tree:
                v = self.game_tree[s_next]['V']
            else:
                v = 0
            q.append(v + r)
            actions.append(a)
        index = secure_random.choice(np.where(q == np.max(q))[0])
        a_state = actions[index]
        s_now = self.game_tree[s_now]['a'][a_state]['s_next']
        return a_state, s_now

    # ToDo: seems like this doesnt do what it is supposed to anymore?
    #  actions do not get saved anywhere....
    # update the policy from the game tree
    def update_policy_from_tree(self, s_0):
        # the idea is to follow a greedy policy from S_0 as long as we can and then switch over to the default rollout policy
        finished = False
        s_now = s_0
        while not finished:
            timestamp, _ = self.decode_states(s_now)
            # do we continue? make sure all terminating conditions are checked for here!
            if timestamp >= self.time_end:
                finished = True

            # do we have Q values for this tree? To do so s_now must be in the tree and have action values
            if s_now in self.game_tree:
                if len(self.game_tree[s_now]['a']) > 0:  # we know the Q values, --> pick greedily
                    a_state, s_now = self.greedy_policy(s_now)

                else:  # well, use the rollout policy then, give a warnign that we ran into a known leaf node (we know the state but not the values)
                    print('using rollout because we found a known leaf node, maybe adjust c_ubc or num_it')
                    _, s_now, a_state, finished = self.one_default_step(s_now=s_now)
            else:  # use rollout policy, we found a totally unknown leaf node! This is potentially bad
                print('using rollout because we found an unknown leaf node, perform troubleshoot pls...')
                _, s_now, a_state, finished = self.one_default_step(s_now=s_now)

            # decoding the actions aleady updates the participants dictionary, not much to do there :-)

            actions = self.decode_actions(a=a_state, timestamp=timestamp)
            self.learner['metrics'][timestamp].update(actions)

    # evaluate current policy of a participant inside a game tree and collects some metrics
    def evaluate_policy(self, do_print=True):
        G = 0
        cumulative_bids_qty = [0, 0]
        cumulative_asks_qty = [0, 0]
        cumulative_grid_buy_qty = 0
        cumulative_grid_sell_qty = 0
        cumulative_financial_buy_qty = 0
        cumulative_financial_sell_qty = 0

        cumulative_bids_price = 0
        cumulative_asks_price = 0

        # cumulative_quantity = 0
        avg_prices = {}
        profile = self.learner['profile']
        # for timestamp in timestamps:
        for step in profile[:-1]:
            timestamp = step['tstamp']
            # return rewards, quantity, metrics
            # r, quantity, avg_price_row = self.get_reward_for_transactions(timestamp)
            # rewards, metrics, bids_qty, asks_qty, grid_transactions[0], grid_transactions[2]
            r, avg_price_row, bids_qty, asks_qty, grid_buy_qty, grid_sell_qty, financial_buy_qty, financial_sell_qty = \
                self.get_reward_for_transactions(timestamp)
            for category in avg_price_row:
                if category not in avg_prices:
                    avg_prices[category] = [avg_price_row[category]]
                else:
                    avg_prices[category].append(avg_price_row[category])
            G += r
            # cumulative_quantity += quantity
            cumulative_bids_qty[1] += bids_qty
            cumulative_asks_qty[1] += asks_qty
            cumulative_grid_buy_qty += grid_buy_qty
            cumulative_grid_sell_qty += grid_sell_qty
            cumulative_financial_buy_qty += financial_buy_qty
            cumulative_financial_sell_qty += financial_sell_qty

            metric_ts = self.learner['metrics'][timestamp]
            metric_ts['exchanged_qty'] = {
                'bids': bids_qty,
                'asks': asks_qty,
                'grid_buy': grid_buy_qty,
                'grid_sell': grid_sell_qty,
                'financial_buy': financial_buy_qty,
                'financial_sell': financial_sell_qty
            }
            # print(metric_ts)
            # print(timestamp, metric_ts)
            # metric_ts = ''

            # metric_ts = self.learner['metrics'][timestamp]
            time_interval = str((timestamp-60, timestamp))
            bid_actions_q = metric_ts['bids'][time_interval]['quantity'] if 'bids' in metric_ts else 0
            ask_actions_q = metric_ts['asks'][time_interval]['quantity'] if 'asks' in metric_ts else 0
            cumulative_bids_qty[0] += bid_actions_q
            cumulative_asks_qty[0] += ask_actions_q

            bid_actions_p = metric_ts['bids'][time_interval]['price'] if 'bids' in metric_ts else 0
            ask_actions_p = metric_ts['asks'][time_interval]['price'] if 'asks' in metric_ts else 0
            cumulative_bids_price += bid_actions_p * bid_actions_q
            cumulative_asks_price += ask_actions_p * ask_actions_q
            # print(bid_actions)

        for category in avg_prices:
            num_nans = np.count_nonzero(np.isnan(avg_prices[category]))
            if num_nans != len(avg_prices[category]):
                avg_prices[category] = np.nanmean(avg_prices[category])
            else:
                avg_prices[category] = np.nan

        if do_print:
            print('Policy of agent ', self.learner_id, ' achieves the following return: ', G)
            # print('actions taken:', [action for action in self.learner['metrics'][timestamp]])

            stats = [
                'quantities (b, a, gb, gs, fb, fs): ',
                str(cumulative_bids_qty[0]) + '|' + str(cumulative_bids_qty[1]),
                str(cumulative_asks_qty[0]) + '|' + str(cumulative_asks_qty[1]),
                cumulative_grid_buy_qty,
                cumulative_grid_sell_qty,
                cumulative_financial_buy_qty,
                cumulative_financial_sell_qty
                ]

            print(*stats)
            print('avg action prices: ',
                  'bids:', round(cumulative_bids_price/cumulative_bids_qty[0], 4) if cumulative_bids_qty[0] > 0 else np.nan,
                  'asks:', round(cumulative_asks_price/cumulative_asks_qty[0], 4) if cumulative_asks_qty[0] > 0 else np.nan
                  )
            print('avg settle prices: ',
                  'bids:', round(avg_prices['avg_bid_price'], 4),
                  'asks:', round(avg_prices['avg_ask_price'], 4))
            print('.........................................')
        return G, cumulative_bids_qty[1] + cumulative_asks_qty[1], avg_prices
