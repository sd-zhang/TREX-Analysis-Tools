from _utils import market_simulation_3b as market
import numpy as np
from _utils import utils
market_configs = {
      "id": "",
      "type": "MicroTE3",
      "close_steps": 2,
      "grid": {
        "price": 0.069,
        "fee_ratio": 1.1
      }
    }

test_market = market.Market(market_configs)
# ledger contains learner's own successful transactions
# market_ledger = [('ask', 10, 0.1, 'solar'),
                 # ('bid', 10, 0.1, 'solar')]

grid_sell_price = market_configs['grid']['price']
grid_buy_price = round(grid_sell_price * (1 + market_configs['grid']['fee_ratio']), 4)
actions = dict()
actions['price'] = sorted(list(np.round(np.linspace(grid_sell_price, grid_buy_price, 9), 5)))
actions['quantity'] = [int(q) for q in list(set(np.floor(np.linspace(-34, 34+1, 7))))]
actions['battery'] = actions['quantity']

for _ in range(1000000):
    price = utils.secure_random.choice(actions['price'])
    quantity = utils.secure_random.choice(actions['quantity'])
    battery = utils.secure_random.choice(actions['battery'])
    load = utils.secure_random.choice([0, 17])
    generation = utils.secure_random.choice([0, 17])

    if quantity < 0:
        market_ledger = [('ask', -quantity, price, 'solar')]
    else:
        market_ledger = [('bid', quantity, price, 'solar')]

    bids, asks, grid_transactions, financial_transactions = test_market.deliver(market_ledger, generation, load, battery)
    if grid_transactions[0] < 0:
        print(quantity, generation, load, battery)
        print(bids, asks, grid_transactions, financial_transactions)
        print('-----')


