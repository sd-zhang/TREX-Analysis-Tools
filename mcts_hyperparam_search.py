# The goal of this is to calculate the emerging nash equilibrium from a given simulation state
# ----------------------------------------------------------------------------------------------------------------------
import copy
import os

import pandas as pd
from _solver.sim_environment import SimulationEnvironment
import numpy as np
import _utils.market_simulation_3b as market

from _utils.rewards_proxy import NetProfit_Reward as Reward
from _utils import utils
from _utils.utils import secure_random
import mcts
from joblib import Parallel, delayed
from _plotter.plotter import log_plotter
from _plotter.plot_policy import policy_plotter
from mcts_nash_solver_3 import Solver
# import matplotlib.pyplot as plt


# ----------------------------------------------------------------------------------------------------------------------



#tobeconsidered hyperparameters:
# generations: more == better, we know this
# iterations_per_gen: more == better, we know this
# c_adjustmen: this will be radically different between BEES and non BEES cases, since BEES makes the tree very tricky
    # take c-adjustment from [1e-4 to 10]
    #
iterations_per_gen = 1000 # we should set this to a reasonably small value, larger = better and we know it.
                            # too large and the hyperparameters wont have an effect
                            # at the same time if it is too small the search will favor broad searches too much!
c_adjustments = [1e-6, 1e-5, 1e-4, 1e-3, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]
if __name__ == '__main__':

    config_name = 'TB3C'
    hyperparameter_search_log = []
    for c_adjustment in c_adjustments:
        hyperparams_search_iterations = []
        for iteration in range(5):
            solver = Solver(config_name)
            log, participants = solver.MA_MCTS(
                max_it_per_gen=iterations_per_gen,
                c_adjustment=c_adjustment,
                learner_fraction_anneal=False,
                hard_reset_game_tree=True,)

            print(solver.study_name)
            # lets extract the important info out here:
            participant_quants = []
            participant_Gs = []
            for participant in log:
                participant_avg_quant = log[participant]['quantity'][:-10]
                participant_avg_quant = np.mean(participant_avg_quant)
                participant_quants.append(participant_avg_quant)
                participant_avg_G = log[participant]['G'][:-10]
                participant_avg_G = np.mean(participant_avg_G)
                participant_Gs.append(participant_avg_G)
            hyperparams_search_iterations.append([participant_avg_G, participant_avg_quant, c_adjustment])

            #setup for folder structure
            folder = solver.study_name
            sub_folder = str(c_adjustment)
            cwd = os.getcwd()
            sub_folder_path = os.path.join(cwd, folder)
            sub_folder_path = os.path.join(sub_folder_path, sub_folder)
            study = 'C' + str(c_adjustment) + '_iteration' +str(iteration)

            # lets make sure we document everything n such
            output = {
                'config': utils.load_config(config_name),
                'metrics': log,
                'participants': participants
            }
            utils.dump_zp(sub_folder_path, study, output)

            # lets generate some plots and save them
            policy_plotter(study_name=study, study_path=sub_folder_path, show_plots=False)
            plotter = log_plotter(output['metrics'],
                                  experiment_name=study,
                                  show_plots=False,
                                  save_plots=True,
                                  folder=sub_folder_path)
            plotter.plot_prices()
            plotter.plot_quantities()
            plotter.plot_returns()

        print(hyperparams_search_iterations)
        c_adjustmen_log = np.mean(hyperparams_search_iterations, axis=0).tolist()
        std_devs = np.std(hyperparams_search_iterations, axis=0).tolist()
        c_adjustmen_log.insert(1, std_devs[0])
        c_adjustmen_log.insert(3, std_devs[1])
        c_adjustmen_log.insert(5, std_devs[2])
        hyperparameter_search_log.append(c_adjustmen_log)
    #process the list:
    hyperparameter_search_log.sort(reverse=True, key=lambda element: (element[0], element[2]))
    utils.dump_zp(folder, 'hyperparam_results_ReturnQuantC', hyperparameter_search_log)
    print('top 4 hyperparameters:', hyperparameter_search_log[:4])

    print('fin')

