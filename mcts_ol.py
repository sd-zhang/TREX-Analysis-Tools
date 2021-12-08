import numpy as np
import copy
from _mcts import node
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
        self.current_timestamp = self.time_start

        self.update_participants(participants)

        actions = self.learner['trader']['actions']
        self.action_spaces = dict()
        self.shape_action_space = list()

        # print(self.shape_action_space)

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

    def init_game_tree(self):
        self.game_tree['root_node'] = node.Node()
        self.game_tree['current_node'] = self.game_tree['root_node']

    # a single step of MCTS, one node evaluation
    def step(self, final_layer):

        # if not self.game_tree['current_node'].children or secure_random.random() <= 0.1:
        #     new_action = secure_random.choice(self.linear_action_space)
        #     if new_action not in self.game_tree['current_node'].children_actions:
        #         self.game_tree['current_node'].add_child(secure_random.choice(self.linear_action_space))
        #         n_next = self.game_tree['current_node'].random_child()

        if not self.game_tree['current_node'].children:
            self.game_tree['current_node'].initialize_children(self.linear_action_space)
            n_next = self.game_tree['current_node'].random_child()
        else:
            n_next = self.game_tree['current_node'].preferred_child_ucb(self.c_adjustment, final_layer)
            # n_next = self.game_tree['current_node'].preferred_child_greedy(final_layer)

        # self.current_timestamp += 60
        # print(self.current_timestamp, final_layer)
        r = self.evaluate_transition(self.current_timestamp, n_next.action) if not final_layer else 0
        self.game_tree['current_node'] = n_next
        # self.game_tree['current_node'].backup_update_jinjerry(r)
        self.game_tree['current_node'].backup_update(r)

    def one_rollout_and_backup(self):
        self.game_tree['current_node'] = self.game_tree['root_node']
        # make sure the root node is created 1 step before the simulation
        # so that the first leaf layer corresponds to first time step
        self.current_timestamp = self.time_start
        while self.current_timestamp < self.time_end:
            final_layer = self.current_timestamp >= self.time_end
            self.step(final_layer)
            self.current_timestamp += 60

    def evaluate_transition(self, timestamp, a):
        actions = self.decode_actions(a=a, timestamp=timestamp)
        self.learner['metrics'][timestamp].update(actions)
        r, _, _, _, _, _, _, _ = self.get_reward_for_transactions(timestamp=timestamp)
        return r

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

    def run(self, reset_tree=False):
        # self.init_game_tree(self.time_start)
        # if not self.game_tree:
        # print(reset_tree)
        if reset_tree:
            print('completely reset game tree')
            self.init_game_tree()
        # else:
        #     print('reset visits for the game tree')
        #     self.reset_visits()
        # s_0 = self.encode_states(time=self.time_start)
        for iteration in range(self.max_iterations):
            self.one_rollout_and_backup()

        self.update_policy_from_tree()
        return {self.learner_id: {
            'game_tree': self.game_tree,
            # 's_0': s_0,
            'metrics': self.learner['metrics']
        }}

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
        for step in profile:
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
            print('Policy of agent ', self.learner_id,
                  ' achieves the following return: ', G)
                  # self.game_tree['root_node'].V/(self.game_tree['root_node'])
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

    def update_policy_from_tree(self):
        self.game_tree['current_node'] = self.game_tree['root_node']
        # print(self.game_tree['current_node'].children)
        timestamp = self.time_start
        while timestamp < self.time_end:
            # n_next = self.game_tree['current_node'].preferred_child_ucb(self.c_adjustment, timestamp >= self.time_end)
            n_next = self.game_tree['current_node'].preferred_child_greedy(timestamp >= self.time_end)
            # print(timestamp)
            actions = self.decode_actions(a=n_next.action, timestamp=timestamp)
            self.learner['metrics'][timestamp].update(actions)
            self.game_tree['current_node'] = n_next
            timestamp += 60