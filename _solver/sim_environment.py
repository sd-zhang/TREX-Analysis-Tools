import commentjson
import numpy as np
import pandas as pd
import dataset
from _utils.Bess import Storage
from _utils import utils

class SimulationEnvironment:
    def __init__(self, config_name):
        self.configs = self.__get_config(config_name)

        study = self.configs['study']
        study['start_timestamp'] = utils.timestr_to_timestamp(study['start_datetime'], study['timezone'])
        study['end_timestamp'] = study['start_timestamp'] + int(study['days'] * 1440) * 60

        print((study['end_timestamp'] - study['start_timestamp'])/60, 'steps')

        self.participants = self.configs['participants']

        for participant in self.participants:
            self.__setup_profiles(participant)
            self.__setup_storage(participant)
            self.__setup_actions(participant)
            # self.__setup_metrics(participant)

    def __get_config(self, config_name: str,):
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
        p = table.find(table.table.columns.tstamp.between(study['start_timestamp'], study['end_timestamp']))
        self.participants[participant]['profile'] = list(p)

    def __setup_actions(self, participant):
        self.participants[participant]['trader']['actions'] = {}

        trader = self.participants[participant]['trader']
        actions = self.participants[participant]['trader']['actions']

        # hard code actions for now. Future versions will utilize config file.
        actions['price'] = tuple(np.linspace(trader['bid_price'], trader['ask_price'], 9))
        actions['quantity'] = tuple(np.arange(11, 25, 2))

        # actions['price'] = tuple(np.array([trader['bid_price'], trader['ask_price']]))
        # actions['quantity'] = tuple(np.array([17]))  # quantity can only be integers

        if 'storage' in self.participants[participant]:
            actions['battery'] = tuple(np.array(([-20, 0, 20])))

    def __setup_storage(self, participant):
        # convert storage params to Storage object
        if 'storage' in self.participants[participant]:
            params = self.participants[participant]['storage']
            self.participants[participant]['storage'] = Storage(**params)

    def __setup_metrics(self, participant):
        self.participants[participant]['metrics'] = {}
        metrics = self.participants[participant]['metrics']
        # format= {'(timestamp_open, timestamp_close)':
        #               'quantity: nbr,
        #               'price': nbr2
        #               'source': string ('solar')
        #               'participant_id': learner
        # if 'storage' in self.participants[participant]:
        #     metrics['soc'] = {}
