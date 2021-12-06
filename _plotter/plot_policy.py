from _utils import utils
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# print(os.getcwd())
def policy_plotter(study_name='study_name_pls', study_path=None, show_plots=False, save_plots=True, autoclose=True):
    #study_name = 'mcts-ute3b-2s-2d-verylong'
    if study_path is None:
        cwd = os.getcwd()
        study_path = os.path.join(cwd, 'logs')
        print(study_path)
    log = utils.import_zp(study_path, study_name)
    study = log['config']['study']
    start_timestamp = utils.timestr_to_timestamp(study['start_datetime'], study['timezone'])
    end_timestamp = start_timestamp + int(study['days'] * 1440) * 60
    steps = int((end_timestamp - start_timestamp) / 60)

    step = 0
    step_size = 60
    time_interval = (start_timestamp + (step - 1) * step_size, start_timestamp + step * step_size)

    policy = {}
    for step in range(steps):
        time_interval = (start_timestamp + (step - 1) * step_size, start_timestamp + step * step_size)
        for participant_id in log['metrics']:
            # if participant_id not in policy:
            #     policy[participant_id] = {}
            history = log['metrics'][participant_id]['history']
            for generation in history:
            # for idx in range(len(history)-500, len(history)):
            #     generation = history[idx]
                actions = generation[time_interval[1]]
                for action_type in ('bids', 'asks'):
                    if action_type in actions:
                        if action_type not in policy:
                            policy[action_type] = list()
                        if str(time_interval) in actions[action_type]:
                            policy[action_type].append({
                                'participant': participant_id,
                                'price': actions[action_type][str(time_interval)]['price'],
                                'quantity': actions[action_type][str(time_interval)]['quantity'],
                                'timestep': step})



    # for participant_id in log['metrics']:
    #     if 'asks' in policy[participant_id]:
    #         plt.hist(policy[participant_id]['asks'][plott], density=True)
    #     if 'bids' in policy[participant_id]:
    #         plt.hist(policy[participant_id]['bids'][plott], density=True)
    #     plt.show()

    category = 'price'
    plt_number = 0
    for action_type in ('bids', 'asks'):
        policy_df = pd.DataFrame(policy[action_type])
        ax = sns.violinplot(x="participant", y=category, hue="timestep", data=policy_df)
        ax.set(title=(' '.join([study_name, action_type, category])))
        if show_plots:
            plt.show()
        if save_plots:
            plot_path = os.path.join(study_path, study_name + "_" + str(plt_number) + '.png' )
            plt.savefig(plot_path)
        if autoclose:
            plt.close()
        plt_number += 1