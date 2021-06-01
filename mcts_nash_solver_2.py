# The goal of this is to calculate the emerging nash equilibrium from a given simulation state
# ----------------------------------------------------------------------------------------------------------------------
import pandas as pd
from _solver.sim_environment import SimulationEnvironment
import numpy as np
import _utils.market_simulation_3 as market
from _plotter.plotter import log_plotter
from _utils.rewards_proxy import NetProfit_Reward as Reward
from _utils.utils import process_profile, secure_random
from _utils.market_simulation_3 import sim_market, _test_settlement_process, _map_market_to_ledger

import matplotlib.pyplot as plt
# ----------------------------------------------------------------------------------------------------------------------
class Solver():
    def __init__(self, config_name, constant_load=True):
        self.simulation_env = SimulationEnvironment(config_name)
        self.reward = Reward()
        for participant in self.simulation_env.participants:
            self.__setup_initial_actions(participant)

    def __setup_initial_actions(self, participant):
        self.simulation_env.participants[participant]['metrics'] = {}
        metrics = self.simulation_env.participants[participant]['metrics']
        # format= {'(timestamp_open, timestamp_close)':
        #               'quantity: nbr,
        #               'price': nbr2
        #               'source': string ('solar')
        #               'participant_id': learner

        # for loop something
        for row in self.simulation_env.participants[participant]['profile']:
            generation, consumption = process_profile(row,
                            gen_scale=self.simulation_env.participants[participant]['generation']['scale'],
                            load_scale=self.simulation_env.participants[participant]['load']['scale'])

            # row['gen'] = gen
            # row['load'] = consumption
            net_load = consumption - generation
            t_start = row['tstamp'] - 60
            t_end = row['tstamp']

            if net_load > 0:
                action_type = 'bids'
            else:
                action_type = 'asks'

            metrics[t_start] = {action_type:
                                    {str((t_start, t_end)):
                                         {'quantity': abs(net_load),
                                         'price': np.random.choice(self.simulation_env.participants[participant]['trader']['actions']['price']),
                                         'source': 'solar',
                                         'participant_id': participant,
                                         },
                                     }
                                }
            if 'storage' in self.simulation_env.participants[participant]:
                metrics[t_start]['battery'] = {'battery_SoC': None, 'target_flux': None}

            metrics[t_start]['gen'] = generation
            metrics[t_start]['load'] = consumption

        return None
    # get the market settlement for one specific row for one specific agent from self.simulation_env.participants
    def _query_market_get_reward_for_one_tuple(self, timestamp, participant,
                                               do_print=False,  # idiot debug flag
                                               ):

        # get the market ledger
        time_start = self.simulation_env.configs['study']['start_timestamp']
        participants = self.simulation_env.participants
        market_df = sim_market(participants=participants,
                               learning_agent_id=participant,
                               timestamp=timestamp)

        market_ledger = []
        quantity = 0

        for index in range(market_df.shape[0]):
            settlement = market_df.iloc[index]
            quantity = settlement['quantity']
            entry = _map_market_to_ledger(settlement, participant, do_print)
            if entry is not None:
                market_ledger.append(entry)
        # if market_ledger:
        #     print(market_ledger)

        # if quantity:
        #     print(quantity)
        # ToDO: test if market is actually doing the right thing

        # we need access to start_energy [0 ... max_energy] and a target_action [-max_energy, max_energy]
        if 'battery' in participants[participant]['metrics'][timestamp]:
            if timestamp == time_start: #FixMe: Apparently Daniel fucked up time here somehow, the very first row of Metrics never gets updated to a real SoC
                bat_SoC_start = 0
            else:
                if timestamp-60 not in participants[participant]['metrics']: # FixMe: catch for general shit
                    print('missing ts!!')
                bat_SoC_start = participants[participant]['metrics'][timestamp-60]['battery']['battery_SoC']

            bat_target_flux = participants[participant]['metrics'][timestamp]['battery']['target_flux']

            # seems like this is error prone somehow?!
            if bat_SoC_start == None: # toDo: catch and fix, once this area is debugged get rid
                print('aha, need to debug')
                bat_SoC_start = 0
            if bat_target_flux == None:
                bat_target_flux = 0

            bat_real_flux, bat_SoC_post = self.simulation_env.participants[self.learner]['storage'].simulate_activity(start_energy=bat_SoC_start, target_energy=bat_target_flux)

            self.simulation_env.participants[learning_participant]['metrics'][timestamp]['battery']['battery_SoC'] = bat_SoC_post
            # if bat_SoC_start - bat_SoC_post  != 0:
            #     print('target flux: ', bat_target_flux)
            #     print('actual flux: ', bat_real_flux)
            #     print('SoC from ', bat_SoC_start, 'to', bat_SoC_post)

        else:
            bat_real_flux = 0

        # calculate the resulting grid transactions
        grid_transactions = self._extract_grid_transactions(market_ledger=market_ledger,
                                                            learning_participant=participant,
                                                            timestamp=timestamp,
                                                            battery=bat_real_flux)
        # print(learning_participant, 'grid trans:', grid_transactions)

        # then calculate the reward function
        rewards, avg_prices = self.reward.calculate(market_transactions=market_ledger,
                                                        grid_transactions=grid_transactions)
        # print('r: ', rewards)
        # if do_print:
        # print('market', market_ledger)
        # print('grid', grid_transactions)
        # print('r', rewards)
        # print('metered_r', simulation_env.participants[learning_participant]['metrics']['reward'][ts])
        return rewards, quantity, avg_prices

    # helper for _query_market_get_reward_for_one_tuple, to see what we get or put into grid
    # ToDo: check here to make sure this is right
    def _extract_grid_transactions(self, market_ledger, learning_participant, timestamp, battery=0.0):
        sucessful_bids = sum([sett[1] for sett in market_ledger if sett[0] == 'bid'])
        sucessful_asks = sum([sett[1] for sett in market_ledger if sett[0] == 'ask'])


        generation = self.simulation_env.participants[learning_participant]['metrics'][timestamp]['gen']
        consumption = self.simulation_env.participants[learning_participant]['metrics'][timestamp]['load']

        residual_consumption = consumption - sucessful_bids
        residual_generation = generation - sucessful_asks
        net_grid_load = residual_consumption - residual_generation

        # not sure what this logically means???
        # commented out for record keeping
        # net_influx = self.simulation_env.participants[learning_participant]['metrics'][timestamp]['gen'] - sucessful_asks
        # net_outflux = self.simulation_env.participants[learning_participant]['metrics'][timestamp]['load'] - sucessful_bids

        # print(learning_participant, market_ledger)
        # print(learning_participant, sucessful_bids, sucessful_asks)
        # print(learning_participant, generation, consumption)
        # print('---')

        grid_load = net_grid_load + battery
        grid_sell_price = self.simulation_env.configs['market']['grid']['price']
        grid_buy_price = grid_sell_price * (1 + self.simulation_env.configs['market']['grid']['fee_ratio'])
        return (max(0, grid_load), grid_buy_price, max(0, -grid_load), grid_sell_price)
        # I think this tweak makes more logical sense?
        # return (max(0, grid_load), max_price, min(0, grid_load), min_price)

    # evaluate current policy of a participant inside a game tree and collects some metrics
    def evaluate_current_policy(self, participant, do_print=True):
        G = 0
        quant_cum = 0
        avg_prices = {}
        # time_start = self.simulation_env.configs['study']['start_timestamp']
        # timestamps = np.arange(time_start, self.time_end, 60)
        profile = self.simulation_env.participants[participant]['profile']
        # for timestamp in timestamps:
        for step in profile[:-1]:
            timestamp = step['tstamp']
            r, quant, avg_price_row = self._query_market_get_reward_for_one_tuple(timestamp, participant, True)

            for category in avg_price_row:
                if category not in avg_prices:
                    avg_prices[category] = [avg_price_row[category]]
                else:
                    avg_prices[category].append(avg_price_row[category])

            G += r
            quant_cum += quant
        for category in avg_prices:
            num_nans = np.count_nonzero(np.isnan(avg_prices[category]))
            if num_nans != len(avg_prices[category]):
                avg_prices[category] = np.nanmean(avg_prices[category])
            else:
                avg_prices[category] = np.nan

        if do_print:
            print('Policy of agent ', participant, ' achieves the following return: ', G)
            print('settled quantity is: ', quant_cum)
            print('avg prices: ', avg_prices)
        return G, quant_cum, avg_prices

    # run MCTS for every agent in the game tree...
    def MA_MCTS(self,
                max_it_per_gen=100,
                c_adjustment=1):
        generations = self.simulation_env.configs['study']['generations']
        log = {}
        game_trees = {}
        s_0s = {}
        action_spaces = {}

        for participant in self.simulation_env.participants:
            log[participant] = {'G': [],
                                'quant': []}

        for gen in range(generations):
            for participant in self.simulation_env.participants:
                print('MCTS gen', gen, 'for', participant)
                game_trees[participant], s_0s[participant], action_spaces[participant] = \
                    self.MCTS(participant, max_it_per_gen, c_adjustment)

            log = self._update_policies_and_evaluate(game_trees, s_0s, action_spaces, log)
        # if self.test_scenario == 'fixed' or self.test_scenario == 'variable':
        #     self._plot_log(log)
        # else:
        #     self._plot_battery()

        return log, game_trees, self.simulation_env.participants

    # one single pass of MCTS for one  learner
    def MCTS(self, learner, max_it, c_adjustment=1):

        # designate the target agent
        # if learner is None:
        #     self.learner = list(self.simulation_env.participants.keys())[0]
        # else:
        self.learner = learner
        print(self.learner)

        # elif self.test_scenario == 'variable' or self.test_scenario == 'fixed' :
        #     self.actions = {'price': np.linspace(self.prices_max_min[1], self.prices_max_min[0], action_space['price']),
        #                     'quantity': np.linspace(0, 30, action_space['quantity'])
        #                     }

        action_space = {}
        self.shape_action_space = []
        actions = self.simulation_env.participants[learner]['trader']['actions']
        for action in actions:
            action_space[action] = len(actions[action])
            self.shape_action_space.append(len(actions[action]))

        # for action_dimension in self.simulation_env.participants[learner]['trader']['actions']:
        #     self.shape_action_space.append(len(self.simulation_env.participants[learner]['trader']['actions'][action_dimension]))
        # determine the size of the action space, I am sure this can be done better

        num_individual_entries = 1
        for dimension in self.shape_action_space:
            num_individual_entries = num_individual_entries*dimension
        self.linear_action_space = np.arange(num_individual_entries).tolist()

        self.c_ucb = c_adjustment

        # determine start and build the actual game tree

        time_start = self.simulation_env.configs['study']['start_timestamp'] #first state of the cropped data piece
        self.time_end = self.simulation_env.configs['study']['end_timestamp'] - 60
        s_0 = self.encode_states(time=time_start - 60)
        game_tree = {}
        game_tree[s_0] = {'N': 0}
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

        # the actual MCTS part
        for it in range(max_it):
            game_tree = self._one_MCT_rollout_and_backup(game_tree, s_0)

        return game_tree, s_0, action_space

    # this update the policy from game tree and evaluate the policy
    def _update_policies_and_evaluate(self, game_trees, s_0s, action_spaces, measurment_dict):

        for participant in self.simulation_env.participants:
            # establish the best policy and test
            game_tree = game_trees[participant]
            s_0 = s_0s[participant]
            action_space = action_spaces[participant]
            self._update_policy_from_tree(participant, game_tree, s_0, action_space)

        for participant in self.simulation_env.participants:
            G, quant, avg_prices = self.evaluate_current_policy(participant=participant, do_print=True)
            if 'G' not in measurment_dict[participant]:
                measurment_dict[participant]['G'] = [G]
            else:
                measurment_dict[participant]['G'].append(G)

            if 'quant' not in measurment_dict[participant]:
                measurment_dict[participant]['quant'] = [quant]
            else:
                measurment_dict[participant]['quant'].append(quant)

            if 'avg_prices' not in measurment_dict[participant]:
                measurment_dict[participant]['avg_prices'] = {}
                for category in avg_prices:
                    measurment_dict[participant]['avg_prices'][category] = [avg_prices[category]]
            else:
                for category in avg_prices:

                    if category not in measurment_dict[participant]['avg_prices']:
                        measurment_dict[participant]['avg_prices'][category] = [avg_prices[category]]
                    else:
                        measurment_dict[participant]['avg_prices'][category].append(avg_prices[category])


        return measurment_dict

    # update the policy from the game tree
    def _update_policy_from_tree(self, participant, game_tree, s_0, action_space):
        # the idea is to follow a greedy policy from S_0 as long as we can and then switch over to the default rollout policy
        finished = False
        s_now = s_0
        while not finished:
            timestamp, _ = self.decode_states(s_now)
            if timestamp <= self.time_end and s_now in game_tree:
                if len(game_tree[s_now]['a']) > 0: #meaning we know the Q values, --> pick greedily

                    Q = []
                    actions = []
                    for a in game_tree[s_now]['a']:
                        r = game_tree[s_now]['a'][a]['r']
                        s_next = game_tree[s_now]['a'][a]['s_next']
                        if s_next in game_tree:
                            V = game_tree[s_next]['V']
                        else:
                            V = 0
                        Q.append(V + r)
                        actions.append(a)

                    index = np.random.choice(np.where(Q == np.max(Q))[0])
                    a_state = actions[index]
                    s_now = game_tree[s_now]['a'][a_state]['s_next']

                else: #well, use the rollout policy then
                    print('using rollout because we found a leaf node, maybe adjust c_ubc or num_it')
                    print(s_now)
                    _, s_now, a_state, finished = self.one_default_step(s_now)

                action_types = [action for action in self.simulation_env.participants[self.learner]['metrics'][timestamp]]
                actions = self.decode_actions(a_state, timestamp, action_types, do_print=True)

            else: #well, use the rollout policy then
                finished = True
                if s_now not in game_tree:
                    print('failed because we found unidentified state!')

    # one MCTS rollout
    def _one_MCT_rollout_and_backup(self, game_tree, s_0):
        s_now = s_0
        action = None
        trajectory = []
        finished = False

        # we're traversing the tree till we hit bottom
        while not finished:
            trajectory.append((s_now, action))
            game_tree, s_now, action, finished = self._one_MCTS_step(game_tree, s_now)

        game_tree = self.bootstrap_values(trajectory, game_tree)

        return game_tree

    # backprop of values
    def bootstrap_values(self, trajectory, game_tree):
        # now we backpropagate the value up the tree:
        if len(trajectory) > 1:
            trajectory.reverse()

        for idx in range(len(trajectory)):
            s_now = trajectory[idx][0]
            # get all possible followup states
            Q = []

            for a in game_tree[s_now]['a']:
                r = game_tree[s_now]['a'][a]['r']
                s_next = game_tree[s_now]['a'][a]['s_next']

                if s_next in game_tree:
                    V_s_next = game_tree[s_next]['V']
                else:
                    V_s_next = 0

                Q.append(r + V_s_next)

            if Q != []:
                game_tree[s_now]['V'] = np.amax(Q)

        return game_tree

    # decode states, placeholder function for more complex states
    def decode_states(self, s):
        timestamp = s[0]
        return timestamp, None

    # same as decode, but backwards...^^
    def encode_states(self, time:int):
        # for now we only encode  time

        if 'battery' in self.simulation_env.participants[self.learner]['trader']['actions']:
            if time+60 <= self.time_start:
                SoC = 0
            else:
                SoC = self.simulation_env.participants[self.learner]['metrics'][time]['battery']['battery_SoC']
        else:
            SoC = None

        t_next = time + 60
        s_next = (t_next, SoC)
        return s_next
    # decode actions, placeholder function for more complex action spaces
    def decode_actions(self, a, ts, action_types, do_print=False):
        actions_dict = self.simulation_env.participants[self.learner]['metrics'][ts]
        # print(actions_dict)
        a = np.unravel_index(int(a), self.shape_action_space)
        # print(price)
        for action_type in action_types:
            if (action_type == 'bids' or action_type == 'asks'):
                actions_dict[action_types[0]] = {str((ts-60, ts)):
                                                {'quantity': self.simulation_env.participants[self.learner]['trader']['actions']['quantity'][a[1]],
                                                'price': self.simulation_env.participants[self.learner]['trader']['actions']['price'][a[0]],
                                                'source': 'solar',
                                                'participant_id': self.learner
                                                }
                                            }
            elif action_type == 'battery':
                actions_dict['battery']['target_flux'] = self.simulation_env.participants[self.learner]['trader']['actions']['battery'][a[-1]]
                actions_dict['battery']['battery_SoC'] = None
        return actions_dict

    # figure out the reward/weight of one transition
    def evaluate_transition(self, s_now, a):
        # for now the state tuple is: (time)
        timestamp, _ = self.decode_states(s_now) # _ being a placeholder for now

        #find the appropriate row in the dataframee
        # row = self.simulation_env.participants[self.learner]['metrics'].index[self.simulation_env.participants[self.learner]['metrics']['timestamp'] == timestamp]
        # row = row[0]

        action_types = [action for action in self.simulation_env.participants[self.learner]['metrics'][timestamp]]
        # print('before: ')
        # print(self.simulation_env.participants[self.learner]['metrics'].at[row, 'actions_dict'])
        actions = self.decode_actions(a, timestamp, action_types)
        # print('after: ')
        # print(self.simulation_env.participants[self.learner]['metrics'].at[row, 'actions_dict'])
        # print(self.simulation_env.participants[self.learner]['metrics']['actions_dict'][row])
        r, _, __ = self._query_market_get_reward_for_one_tuple(timestamp, self.learner, do_print=False)
        s_next = self.encode_states(time=timestamp)

        # print(r)
        return r, s_next

    # determine the next stat
    def _next_states(self, s_now, a):
        t_now = s_now[0]
        s_next = self.encode_states(time=t_now)
        return s_next

    # a single step of MCTS, one node evaluation
    def _one_MCTS_step(self, game_tree, s_now):
        #see if wee are in a leaf node
        finished = False

        # check of leaf node, if leaf node then do rollout, estimate V of node
        if 'a' not in game_tree[s_now]:
            game_tree[s_now]['V'] = self.default_rollout(s_now)
            game_tree[s_now]['a'] = {}
            game_tree[s_now]['N'] += 0

            finished = True
            s_next = None
            a = None

        # its no leaf node, so we expand using ucb policy
        else:

            a = self._ucb(game_tree, s_now)
            if a not in game_tree[s_now]['a']: #equivalent to game_tree[s_now]['a'][a]['n'] == 0
                game_tree[s_now]['a'][a] = {'r': None,
                                            'n': 0,
                                            's_next': None} #gotta mak sure all those get populated

            r, s_next = self.evaluate_transition(s_now, a)
            ts, _ = self.decode_states(s_next)
            game_tree[s_now]['a'][a]['r'] = r
            game_tree[s_now]['a'][a]['n'] += 1
            game_tree[s_now]['N'] += 1
            game_tree[s_now]['a'][a]['s_next'] = s_next

            # update V estimate for node
            if s_next not in game_tree and ts <= self.time_end:
                game_tree[s_next] = {'N': 0}

            if ts > self.time_end:
                 finished = True

        return game_tree, s_next, a, finished
            # else:
            #
            #     r, s_next = self.evaluate_transition(s_now, a)
            #
            #     game_tree[s_now]['a'][a]['r'] = r
            #    game_tree[s_now]['a'][a]['s_next'] = s_next
            #     game_tree[s_now]['a'][a]['n'] +=1
            #     game_tree[s_now]['N'] += 1
            #     ts_now, _ = self.decode_states(s_now)
            #     if ts_now >= self.time_end:
            #         finished = True
            #
            #     return game_tree, s_next, finished

    # add to the game tree
    def __add_s_next(self, game_tree, s_now, action_space):
        a_next = {}
        for action in action_space:

            s_next = self._next_states(s_now, action)
            ts, _ = self.decode_states(s_next)
            if s_next not in game_tree and ts <= self.time_end:
                game_tree[s_next] = {'N': 0}

            a_next[str(action)] = {'r': None,
                              'n': 0,
                              's_next': s_next}


        game_tree[s_now]['a'] = a_next

        return game_tree

    # here's the two policies that we'll be using for now:
    # UCB for the tree traversal
    # random action selection for rollouts
    def one_default_step(self, s_now):
        finished = False
        a = np.random.choice(self.linear_action_space)
        ts, _ = self.decode_states(s_now)
        r, s_next = self.evaluate_transition(s_now, a)
        if ts == self.time_end:
            finished = True

        return r, s_next, a, finished

    def default_rollout(self, s_now):
        finished = False
        V = 0
        while not finished:
            r, s_next, _, finished = self.one_default_step(s_now)
            s_now = s_next
            V += r

        return V

    def _ucb(self, game_tree, s_now, c=0.05):
        # UCB formula: V_ucb_next = V + c*sqrt(ln(N_s)/n_s_next)

        N_s = game_tree[s_now]['N']
        all_s_next = []
        num_actions = len(self.linear_action_space)
        Q_ucb = [None]*num_actions # determine the value of all followup transitions states

        for idx_a in range(num_actions):
            a = self.linear_action_space[idx_a]
            if a not in game_tree[s_now]['a']:
                n_next = 0 #if the aqction transition isnt logged, we havent sampled it yet
                Q_ucb[idx_a] = np.inf
            else:
                s_next = game_tree[s_now]['a'][a]['s_next']
                n_next = game_tree[s_now]['a'][a]['n']

                r = game_tree[s_now]['a'][a]['r']
                ts, _ = self.decode_states(s_next)
                if ts >= self.time_end:
                    V_next = 0
                else:
                    V_next = game_tree[s_next]['V']
                Q = r + V_next
                Q_ucb[idx_a]= Q + self.c_ucb*np.sqrt(np.log(N_s)/n_next)

        #making sure we pick the maximums at random
        a_ucb_index = np.random.choice(np.where(Q_ucb == np.max(Q_ucb))[0])
        a_ucb = self.linear_action_space[a_ucb_index]
        return a_ucb


if __name__ == '__main__':
    solver = Solver('TB3T', constant_load=True)
    log, game_trees, participants_dict = solver.MA_MCTS(max_it_per_gen=1000)
    plotter = log_plotter(log)
    # plotter.plot_prices()
    plotter.plot_quantities()
    # plotter.plot_returns()
    log_plotter(log)
    print('fin')