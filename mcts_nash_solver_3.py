# The goal of this is to calculate the emerging nash equilibrium from a given simulation state
# ----------------------------------------------------------------------------------------------------------------------
import copy

import pandas as pd
from _solver.sim_environment import SimulationEnvironment
import numpy as np
import _utils.market_simulation_3b as market
from _plotter.plotter import log_plotter
from _utils.rewards_proxy import NetProfit_Reward as Reward
from _utils.utils import process_profile, secure_random
import mcts
from joblib import Parallel, delayed
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------------------------------------------------------
class Solver:
    def __init__(self, config_name):
        self.simulation_env = SimulationEnvironment(config_name)
        # self.participants = self.simulation_env.participants
        self.reward = Reward()
        self.market = market.Market(self.simulation_env.configs['market'])
        self.metrics = dict()

    def update_metrics(self, participant, G, quantity, avg_prices):
        # format log
        if 'G' not in self.metrics[participant]:
            self.metrics[participant]['G'] = [G]
        else:
            self.metrics[participant]['G'].append(G)
        if 'quantity' not in self.metrics[participant]:
            self.metrics[participant]['quantity'] = [quantity]
        else:
            self.metrics[participant]['quantity'].append(quantity)
        if 'avg_prices' not in self.metrics[participant]:
            self.metrics[participant]['avg_prices'] = {}
            for category in avg_prices:
                self.metrics[participant]['avg_prices'][category] = [avg_prices[category]]
        else:
            for category in avg_prices:
                if category not in self.metrics[participant]['avg_prices']:
                    self.metrics[participant]['avg_prices'][category] = [avg_prices[category]]
                else:
                    self.metrics[participant]['avg_prices'][category].append(avg_prices[category])

# run MCTS for every agent in the game tree...
    def MA_MCTS(self,
                max_it_per_gen,
                c_adjustment,
                learner_fraction_anneal=False, #Experimental feature that might help calm violence of conversion
                ):
        generations = self.simulation_env.configs['study']['generations']
        learning_participants = [participant for participant in self.simulation_env.participants if
                                 self.simulation_env.participants[participant]['trader']['learning']]

        # active_learning_participants = learning_participants

        for participant in learning_participants:
            self.metrics[participant] = {'G': list(),
                                         'quantity': list()}

        learning_mcts = dict()
        for participant_id in learning_participants:
            learning_mcts[participant_id] = mcts.MCTS(
                participants=self.simulation_env.participants,
                learner_id=participant_id,
                market=self.market,
                reward=self.reward,
                time_start=self.simulation_env.configs['study']['start_timestamp'],
                time_end=self.simulation_env.configs['study']['end_timestamp'] - 60,
                max_iterations=max_it_per_gen,
                c_adjustment=c_adjustment
            )

        for gen in range(generations):
            if learner_fraction_anneal:
                fraction_to_optimize = (generations - gen)/generations
                fraction_to_optimize = len(learning_participants) * fraction_to_optimize
                fraction_to_optimize = int(np.ceil(fraction_to_optimize))

                if fraction_to_optimize > 1:
                    active_learning_participants = secure_random.sample(learning_participants, fraction_to_optimize)
                else:
                    learning_participants = list(np.roll(learning_participants, 1))
                    active_learning_participants = [learning_participants[0]]

                print(active_learning_participants)
                print('selecting ', fraction_to_optimize/len(learning_participants)*100, 'percent of available participants to learn')
            else:
                active_learning_participants = learning_participants

            for participant_id in learning_participants:
                learning_mcts[participant_id].update_participants(self.simulation_env.participants)

            # serial execution code
            # for participant_id in active_learning_participants:
            #     print('MCTS gen', gen, 'for', participant_id)
            #     result = learning_mcts[participant_id].run()
            #     learning_mcts[participant_id].update_policy_from_tree(result[participant_id]['s_0'])

            # parallel execution code
            print('MCTS gen', gen)
            with Parallel(n_jobs=len(active_learning_participants)) as parallel:
                results = parallel(delayed(learning_mcts[participant_id].run)() for
                                   participant_id in active_learning_participants)
            for result in results:
                for participant_id in result:
                    learning_mcts[participant_id].game_tree.update(result[participant_id]['game_tree'])
                    learning_mcts[participant_id].learner['metrics'].update(result[participant_id]['metrics'])
                    learning_mcts[participant_id].update_policy_from_tree(result[participant_id]['s_0'])

            for participant_id in learning_participants:
                G, cumulative_quantity, avg_prices = learning_mcts[participant_id].evaluate_policy()
                self.update_metrics(participant_id, G, cumulative_quantity, avg_prices)
                self.simulation_env.participants[participant_id]['metrics'].update(
                    learning_mcts[participant_id].learner['metrics'])

        return self.metrics, self.simulation_env.participants

if __name__ == '__main__':
    solver = Solver('TB3B')
    log, participants_dict = solver.MA_MCTS(max_it_per_gen=1000, c_adjustment=1, learner_fraction_anneal=True)

    plotter = log_plotter(log)
    plotter.plot_prices()
    plotter.plot_quantities()
    plotter.plot_returns()
    log_plotter(log)
    print('fin')