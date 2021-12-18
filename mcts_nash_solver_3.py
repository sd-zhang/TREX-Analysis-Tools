# The goal of this is to calculate the emerging nash equilibrium from a given simulation state
# ----------------------------------------------------------------------------------------------------------------------
import copy

import pandas as pd
from _solver.sim_environment import SimulationEnvironment
import numpy as np
import _utils.market_simulation_3b as market

from _utils.rewards_proxy import NetProfit_Reward as Reward
from _utils import utils
from _utils.utils import secure_random
import mcts
# import mcts_ol as mcts
from joblib import Parallel, delayed
from _plotter.plotter import log_plotter
from _plotter.plot_policy import policy_plotter
# import matplotlib.pyplot as plt


# ----------------------------------------------------------------------------------------------------------------------
class Solver:
    def __init__(self, config_name):
        self.simulation_env = SimulationEnvironment(config_name)
        self.study_name = self.simulation_env.configs['study']['name']
        # self.participants = self.simulation_env.participants
        self.reward = Reward()
        self.market = market.Market(self.simulation_env.configs['market'])
        self.metrics = dict()

    def _metrics_append(self, participant, key, entry):
        if key not in self.metrics[participant]:
            self.metrics[participant][key] = [entry]
        else:
            self.metrics[participant][key].append(entry)

    def update_metrics(self, participant,
                        G=None,
                        G_expected=None,
                        quantity=None,
                        quantity_expected=None,
                        avg_prices=None,
                        avg_prices_expected=None,
                        metrics_history=None):
        # format log
        if G is not None:
            self._metrics_append(participant, 'G', G)

        if G_expected is not None:
            self._metrics_append(participant, 'G_expected', G_expected)

        if quantity is not None:
            self._metrics_append(participant, 'quantity', quantity)

        if quantity_expected is not None:
            self._metrics_append(participant, 'quantity_expected', quantity_expected)

        if avg_prices is not None:
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

        if metrics_history is not None:
            self._metrics_append(participant, 'history', metrics_history)

# run MCTS for every agent in the game tree...
    def MA_MCTS(self,
                max_it_per_gen,
                c_adjustment,
                hard_reset_game_tree=False,
                ):
        generations = self.simulation_env.configs['study']['generations']
        learning_participants = [participant for participant in self.simulation_env.participants if
                                 self.simulation_env.participants[participant]['trader']['learning']]

        # active_learning_participants = learning_participants

        for participant in learning_participants:
            self.metrics[participant] = {'G': list(),
                                         'quantity': list()}

        game_trees = dict()
        learning_mcts = dict()
        for participant_id in learning_participants:
            learning_mcts[participant_id] = mcts.MCTS(
                participants=self.simulation_env.participants,
                learner_id=participant_id,
                market=self.market,
                reward=self.reward,
                time_start=self.simulation_env.configs['study']['start_timestamp'],
                time_end=self.simulation_env.configs['study']['end_timestamp'],
                max_iterations=max_it_per_gen,
                c_adjustment=c_adjustment
            )

        for gen in range(generations):
            for participant_id in learning_participants:
                learning_mcts[participant_id].update_participants(self.simulation_env.participants)

            # serial execution code
            # for participant_id in active_learning_participants:
            #     print('MCTS gen', gen, 'for', participant_id)
            #     result = learning_mcts[participant_id].run()
            #     learning_mcts[participant_id].update_policy_from_tree(result[participant_id]['s_0'])

            # parallel execution code
            #ToDo: follow this a little bit to see how this works now
            print('MCTS gen', gen)
            reset_tree = False
            prune = False
            if gen <= 0 or hard_reset_game_tree: #might wanna comment this out
                reset_tree = True

            # if not reset_tree and gen > int(generations//5):
            # if not reset_tree and gen:
            #     prune = True

            with Parallel(n_jobs=len(learning_participants)) as parallel:
                results = parallel(delayed(learning_mcts[participant_id].run)(reset_tree) for
                                   participant_id in learning_participants)

            for result in results:
                for participant_id in result:
                    # copy tree and metrics back into MCTS instances to deal with parallel processing oddity
                    learning_mcts[participant_id].game_tree.update(result[participant_id]['game_tree'])
                    learning_mcts[participant_id].learner['metrics'].update(result[participant_id]['metrics'])

                    G_expected, cumulative_quantity_expected, avg_prices_expected = learning_mcts[participant_id].evaluate_policy()
                    self.update_metrics(participant_id,
                                        G_expected=G_expected,
                                        quantity_expected=cumulative_quantity_expected,
                                        avg_prices_expected=avg_prices_expected,
                                        metrics_history=learning_mcts[participant_id].learner['metrics'])
                    self.simulation_env.participants[participant_id]['metrics'].update(
                        learning_mcts[participant_id].learner['metrics'])

                    game_trees[participant_id] = learning_mcts[participant_id].game_tree

            # To determine regret
            for participant_id in learning_participants:
                G_achieved, cumulative_quantity_achieved, avg_prices_achieved = learning_mcts[participant_id].evaluate_policy()
                self.update_metrics(participant_id,
                                    G=G_achieved,
                                    quantity=cumulative_quantity_achieved,
                                    avg_prices=avg_prices_achieved,)
        return self.metrics, self.simulation_env.participants, game_trees

if __name__ == '__main__':
    config_name = 'TB6'
    solver = Solver(config_name)
    log, participants, game_trees = solver.MA_MCTS(
        max_it_per_gen=20000,
        c_adjustment=1,
        hard_reset_game_tree=True)
    print(solver.study_name)
    output = {
        'config': utils.load_config(config_name),
        'metrics': log,
        'participants': participants,
        'game_trees': game_trees
    }

    utils.dump_zp('logs', solver.study_name, output)
    policy_plotter(study_name=solver.study_name)

    plotter = log_plotter(output['metrics'], experiment_name='Hard Tree Resets 1000Its 1000Gens TB6')
    plotter.plot_prices()
    plotter.plot_quantities()
    plotter.plot_returns()
    log_plotter(log)
    print('fin')


# Done, needs tests: make sure we also do this in our MCTS, at the end of run call update_policy_from_tree
# ToDo: update decode actions simialrly to Steven's MCTS
# Done, needs tests: make a dict comprehension or subfunction to clear metrics dictionary after every decode acctions call, similar to Steven's MCTS
# ToDo: check timestamps to be consistent with steven's code, check game-tree to see if it does what it is supposed to
# ToDo: update plotter to plot the expected return or the regret...

# test stuff, unneded
# from _utils import utils
# log = utils.import_zp("D:/TREX/TREX-Analysis-Tools/mcts-ute3b-bess-2s/0.1/", "C0.1_iteration0")
# log = utils.import_zp("logs/", "mcts-ute3b-bess-1s-1d-soft-reset-random")