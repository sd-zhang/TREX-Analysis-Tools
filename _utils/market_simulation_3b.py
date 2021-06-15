from operator import itemgetter
import itertools
import copy
import pandas as pd
import numpy as np

class Market:
    def __init__(self, configs):
        self.configs = configs
        self.grid_sell_price = self.configs['grid']['price']
        self.grid_buy_price = self.grid_sell_price * (1 + self.configs['grid']['fee_ratio'])

    def match(self, bids, asks, time_delivery):
        settled = []
        bids = sorted([bid for bid in bids if (bid['quantity'] > 0)], key=itemgetter('price'), reverse=True)
        asks = sorted([ask for ask in asks if (ask['quantity'] > 0)], key=itemgetter('price'), reverse=False)

        for bid, ask, in itertools.product(bids, asks):
            if ask['price'] > bid['price']:
                continue

            if bid['participant_id'] == ask['participant_id']:
                continue

            if bid['quantity'] <= 0 or ask['quantity'] <= 0:
                continue

            # Settle highest price bids with lowest price asks
            settle_record = self.settle(bid, ask, time_delivery)
            # print(settle_record)
            if settle_record:
                settled.append(settle_record)
                bid['quantity'] -= settle_record['quantity']
                ask['quantity'] -= settle_record['quantity']
        return settled

    def settle(self, bid: dict, ask: dict, time_delivery: tuple):
        # only proceed to settle if settlement quantity is positive
        quantity = min(bid['quantity'], ask['quantity'])
        if quantity <= 0:
            return

        settlement_price_sell = ask['price']
        settlement_price_buy = bid['price']

        record = {
            'quantity': quantity,
            'seller_id': ask['participant_id'],
            'buyer_id': bid['participant_id'],
            'energy_source': ask['source'],
            'settlement_price_sell': settlement_price_sell,
            'settlement_price_buy': settlement_price_buy,
            'time_delivery': time_delivery
        }
        return record

    def deliver(self, market_ledger, generation, consumption, battery=0, ):
        # sort asks from highest to lowest
        # sort bids from lowest to highest
        bids = sorted([sett for sett in market_ledger if sett[0] == 'bid'], key=lambda x: x[2], reverse=True)
        asks = sorted([sett for sett in market_ledger if sett[0] == 'ask'], key=lambda x: x[2], reverse=False)

        total_bids = sum([bid[1] for bid in bids])
        total_asks = sum([ask[1] for ask in asks])

        grid_buy = 0
        grid_sell = 0

        financial_buy = [0, 0]
        financial_sell = [0, 0]

        net_consumption = consumption - generation
        # separate bids into physical and financial
        if total_bids > net_consumption:
            bid_deficit = total_bids - (max(0, net_consumption) + battery)
            if bid_deficit > 0:
                while bid_deficit:
                    # print(learning_participant, total_bids, net_consumption, battery, bid_deficit)
                    for idx in range(len(bids)):
                        bid = list(bids[idx])
                        compensation = min(bid_deficit, bid[1])
                        financial_buy[0] += compensation
                        financial_buy[1] += compensation * bid[2]
                        bid[1] -= compensation
                        bid_deficit -= compensation
                        bids[idx] = tuple(bid)
            else:
                grid_buy -= bid_deficit
        elif total_bids < net_consumption:
            residual_consumption = net_consumption - total_bids
            if battery <= 0:
                residual_battery = -battery - residual_consumption
                if residual_battery > 0:
                    grid_sell += residual_battery
                else:
                    grid_buy += residual_consumption + battery
            else:
                grid_buy += residual_consumption + battery

        # sell more than generated
        if total_asks > generation:
            deficit_generation = total_asks - generation
            # print(deficit_generation)
            # if battery discharging
            if battery <= 0:
                if -battery > deficit_generation:
                    residual_battery = -battery - deficit_generation
                    deficit_generation = 0
                    if residual_battery > consumption:
                        grid_sell += residual_battery - consumption
                    else:
                        grid_buy = consumption - residual_battery
                else:
                    deficit_generation += battery
            while deficit_generation:
                for idx in range(len(asks)):
                    ask = list(asks[idx])
                    compensation = min(deficit_generation, ask[1])
                    financial_buy[0] += compensation
                    financial_buy[1] += compensation * self.grid_buy_price
                    ask[1] -= compensation
                    deficit_generation -= compensation
                    asks[idx] = tuple(ask)
            # if battery charging or doing nothing
            else:
                financial_buy[0] += deficit_generation
                financial_buy[1] += deficit_generation * self.grid_buy_price
                grid_buy += battery
                grid_buy += consumption

        # if sell less than generated
        elif total_asks < generation:
            residual_generation = generation - total_asks
            if residual_generation > consumption:
                grid_sell += (residual_generation - consumption)
            elif residual_generation < consumption:
                grid_buy += (consumption - residual_generation)

        grid_transactions = (grid_buy, self.grid_buy_price, grid_sell, self.grid_sell_price)
        return bids, asks, grid_transactions, financial_buy + financial_sell


    # simulated market for participants, giving back learning agent's settlements, optionally for a specific timestamp
    def simulate_transactions(self, participants: dict, learner_id:str, timestamp:int):
        # learning_agent = participants[learner_id]
        # # opponents = copy.deepcopy(participants)
        # # opponents.pop(learning_agent_id, None)
        # # print(learning_agent_id)
        open = dict()
        learning_agent_times_delivery = list()
        transactions_df = list()
        timestamps = [timestamp]
        # get all actions taken by all agents for a time interval
        for ts in timestamps:
            for participant_id in participants:
                agent_actions = participants[participant_id]['metrics'][ts]
                # print(agent_actions)
                for action in ('bids', 'asks'):
                    if action in agent_actions:
                        for time_delivery in agent_actions[action]:
                            # print(time_delivery)
                            if time_delivery not in open:
                                open[time_delivery] = {}
                            if action not in open[time_delivery]:
                                open[time_delivery][action] = []

                            aa = agent_actions[action][time_delivery]
                            aa['participant_id'] = participant_id
                            open[time_delivery][action].append(copy.deepcopy(aa))
                            if participant_id == learner_id:
                                learning_agent_times_delivery.append(time_delivery)

        for t_d in learning_agent_times_delivery:
            if 'bids' in open[t_d] and 'asks' in open[t_d]:
                transactions_df.extend(self.match(open[t_d]['bids'], open[t_d]['asks'], t_d))
        return pd.DataFrame(transactions_df)


    # all the stuff we need to calculate returns
    def simulated_transactions_to_ledger(self, market_df_ts, learner_id):
        quantity = market_df_ts['quantity']
        source = market_df_ts['energy_source']
        ledger_entry = None

        if market_df_ts['seller_id'] == learner_id or market_df_ts['buyer_id'] == learner_id:
            if learner_id == market_df_ts['seller_id']:
                action = 'ask'
                price = market_df_ts['settlement_price_sell']
            elif learner_id == market_df_ts['buyer_id']:
                action = 'bid'
                price = market_df_ts['settlement_price_buy']
            else:
                action = None
                price = None
            ledger_entry = (action, quantity, price, source)
        return ledger_entry
