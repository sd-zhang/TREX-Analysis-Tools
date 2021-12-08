import commentjson
import numpy as np
import pandas as pd
import dataset
from _utils.Bess import Storage
from _utils import utils
from _utils.utils import process_profile, secure_random

class SimulationEnvironment:
    def __init__(self, config_name):
        self.configs = self.__get_config(config_name)
        study = self.configs['study']
        study['start_timestamp'] = utils.timestr_to_timestamp(study['start_datetime'], study['timezone'])
        study['end_timestamp'] = study['start_timestamp'] + int(study['days'] * 1440) * 60

        print(study['name'], study['start_timestamp'], study['end_timestamp'])
        print((study['end_timestamp'] - study['start_timestamp']) / 60, 'steps')

        self.participants = self.configs['participants']

        for participant in self.participants:
            self.__setup_profiles(participant)
            self.__setup_storage(participant)
            self.__setup_actions(participant)
            # self.__setup_metrics(participant)
            self.__setup_initial_actions(participant)

    def __get_config(self, config_name: str):
        config_file = '_configs/' + config_name + '.json'
        # config_file = 'E:/TREX-Analysis-Tools/_configs/TB3C.json'
        with open(config_file) as f:
            config = commentjson.load(f)
        return config

    def __setup_profiles(self, participant):
        db = dataset.connect(self.configs['study']['profiles_db_location'])
        study = self.configs['study']
        trader = self.participants[participant]['trader']
        table_name = trader['use_synthetic_profile'] if 'use_synthetic_profile' in trader else participant
        table = db[table_name]
        # profile timestamp is the end of the measurement interval
        p = table.find(table.table.columns.tstamp.between(study['start_timestamp'], study['end_timestamp']))
        self.participants[participant]['profile'] = list(p)[:-1]
        # print(self.participants[participant]['profile'][0])

    def __setup_actions(self, participant):
        trader = self.participants[participant]['trader']
        if 'actions' not in trader:
            trader['actions'] = {}
        actions = trader['actions']

        # crawl through the profile and figure out the maximum quantity to avoid too many "equivalent" states
        max_generation = 0
        max_consumption = 0
        for row in self.participants[participant]['profile']:
            generation, consumption = process_profile(row,
                            gen_scale=self.participants[participant]['generation']['scale'],
                            load_scale=self.participants[participant]['load']['scale'])
            max_generation = max(max_generation, generation)
            max_consumption = max(max_consumption, consumption)
        max_qty = max(max_generation, max_consumption)

        # hard code actions for now. Future versions will utilize config file.
        if 'price' not in actions or not actions['price']:
            # actions['price'] = list(np.linspace(trader['bid_price'], trader['ask_price'], 15))
            # actions['price'] = list(np.round(np.linspace(0.05, 0.16, 15), 5))

            # limit prices to be between grid
            grid_price_sell = self.configs['market']['grid']['price']
            grid_price_buy = grid_price_sell * (1 + self.configs['market']['grid']['fee_ratio'])
            actions['price'] = sorted(list(np.round(np.linspace(grid_price_sell, grid_price_buy, 9), 5)))

        if 'quantity' not in actions or not actions['quantity']:
            actions['quantity'] = list(range(0, max_qty+1, 1))
            # actions['quantity'] = list(range(0, max_qty, 3)) + [max_qty]

        # actions['price'] = tuple(np.linspace(trader['bid_price'], trader['ask_price']imp, 3))
        # actions['price'] = tuple(np.array([0.1]))
        # actions['quantity'] = tuple(np.array([17]))  # quantity can only be integers

        # print(participant, actions)

        if 'storage' in self.participants[participant]:
            if 'battery' not in actions or not actions['battery']:
                # actions['battery'] = tuple(range(-20, 20, 1))
                # actions['battery'] = list(range(-max_qty, max_qty+1, 3))
                actions['battery'] = list(set(list(-np.array(actions['quantity'])) + actions['quantity']))
                # temporarily disable price and quantity to speed up self consumption sim
                # actions['price'] = [0]
                # actions['quantity'] = [0]

    def __setup_storage(self, participant):
        # convert storage params to Storage object
        if 'storage' in self.participants[participant]:
            params = self.participants[participant]['storage']
            self.participants[participant]['storage'] = Storage(**params)

    def __setup_metrics(self, participant):
        if 'metrics' not in self.participants[participant]:
            self.participants[participant]['metrics'] = {}
        # metrics = self.participants[participant]['metrics']
        # format= {'(timestamp_open, timestamp_close)':
        #               'quantity: nbr,
        #               'price': nbr2
        #               'source': string ('solar')
        #               'participant_id': learner
        # if 'storage' in self.participants[participant]:
        #     metrics['soc'] = {}

    def __setup_initial_actions(self, participant):
        if 'metrics' not in self.participants[participant]:
            self.participants[participant]['metrics'] = {}

        metrics = self.participants[participant]['metrics']
        # format= {'(timestamp_open, timestamp_close)':
        #               'quantity: nbr,
        #               'price': nbr2
        #               'source': string ('solar')
        #               'participant_id': learner

        actions = self.participants[participant]['trader']['actions']
        # for loop something
        for row in self.participants[participant]['profile']:
            generation, consumption = process_profile(row,
                            gen_scale=self.participants[participant]['generation']['scale'],
                            load_scale=self.participants[participant]['load']['scale'])

            # row['gen'] = gen
            # row['load'] = consumption
            net_load = consumption - generation
            t_start = row['tstamp'] - 60
            t_end = row['tstamp']

            # print(participant, t_end, generation, consumption)
            if t_end not in metrics:
                metrics[t_end] = dict()

            target_flux = 0
            if 'storage' in self.participants[participant]:
                target_flux = secure_random.choice(actions['battery'])
                net_load += target_flux

            if net_load >= 0:
                action_type = 'bids'
                # print(participant, t_start, generation, consumption, net_load, action_type)
                metrics[t_end][action_type] = {str((t_start, t_end)):
                                                   {'quantity': secure_random.choice(actions['quantity']),
                                                    'price': secure_random.choice(actions['price'][len(actions['price'])//2:]),
                                                    'participant_id': participant,
                                                    }
                                               }
            else:
                action_type = 'asks'
                metrics[t_end][action_type] = {str((t_start, t_end)):
                                                   {'quantity': secure_random.choice(actions['quantity']),
                                                    'price': secure_random.choice(actions['price'][:len(actions['price'])//2]),
                                                    'source': 'solar',
                                                    'participant_id': participant,
                                                    }
                                               }

            if 'storage' in self.participants[participant]:
                metrics[t_end]['battery'] = {'battery_SoC': None,
                                             'target_flux': target_flux}
                                             # 'target_flux': secure_random.choice(actions['battery'])}
                                             # 'target_flux': None}

            # print(participant, t_end, metrics)

            metrics[t_end]['gen'] = generation
            metrics[t_end]['load'] = consumption
