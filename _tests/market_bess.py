from _utils import market_simulation_3b as market
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

# case 1: perfect bids for consumer, discharge battery
market_ledger = [('bid', 10, 0.1, 'solar')]
bids, asks, grid_transactions, financial_transactions = test_market.deliver(market_ledger, 0, 10, -10)

check = [bids == [('bid', 10, 0.1, 'solar')],
         asks == [],
         grid_transactions == (0, 0.1449, 10, 0.069),
         financial_transactions == [0, 0, 0, 0]]
print('c1', check)
print(bids, asks, grid_transactions, financial_transactions)
print('-----')

# case 2: perfect bids for consumer, charge battery
market_ledger = [('bid', 10, 0.1, 'solar')]
bids, asks, grid_transactions, financial_transactions = test_market.deliver(market_ledger, 0, 10, 10)

check = [bids == [('bid', 10, 0.1, 'solar')],
         asks == [],
         grid_transactions == (10, 0.1449, 0, 0.069),
         financial_transactions == [0, 0, 0, 0]]
print('c2', check)
print(bids, asks, grid_transactions, financial_transactions)
print('-----')

# case 3: bid more than consumption for consumer, discharge battery
market_ledger = [('bid', 20, 0.1, 'solar')]
bids, asks, grid_transactions, financial_transactions = test_market.deliver(market_ledger, 0, 10, -10)

check = [bids == [('bid', 10, 0.1, 'solar')],
         asks == [],
         grid_transactions == (0, 0.1449, 10, 0.069),
         financial_transactions == [10, 1.0, 0, 0]]
print('c3', check)
print(bids, asks, grid_transactions, financial_transactions)
print('-----')

# case 4: bid more than consumption for consumer, charge battery
market_ledger = [('bid', 20, 0.1, 'solar')]
bids, asks, grid_transactions, financial_transactions = test_market.deliver(market_ledger, 0, 10, 10)

check = [bids == [('bid', 20, 0.1, 'solar')],
         asks == [],
         grid_transactions == (0, 0.1449, 0, 0.069),
         financial_transactions == [0, 0, 0, 0]]
print('c4', check)
print(bids, asks, grid_transactions, financial_transactions)
print('-----')

# case 5: bid less than consumption for consumer, discharge battery
market_ledger = [('bid', 5, 0.1, 'solar')]
bids, asks, grid_transactions, financial_transactions = test_market.deliver(market_ledger, 0, 10, -10)
# expected output:
check = [bids == [('bid', 5, 0.1, 'solar')],
         asks == [],
         grid_transactions == (0, 0.1449, 5, 0.069),
         financial_transactions == [0, 0, 0, 0]]
print('c5', check)
print(bids, asks, grid_transactions, financial_transactions)
print('-----')

# case 6: bid less than consumption for consumer, charge battery
market_ledger = [('bid', 5, 0.1, 'solar')]
bids, asks, grid_transactions, financial_transactions = test_market.deliver(market_ledger, 0, 10, 10)
# expected output:
check = [bids == [('bid', 5, 0.1, 'solar')],
         asks == [],
         grid_transactions == (15, 0.1449, 0, 0.069),
         financial_transactions == [0, 0, 0, 0]]
print('c6', check)
print(bids, asks, grid_transactions, financial_transactions)
print('-----')

# case 7: perfect asks for generator, discharge battery
market_ledger = [('ask', 10, 0.1, 'solar')]
bids, asks, grid_transactions, financial_transactions = test_market.deliver(market_ledger, 10, 0, -10)
# expected output:
check = [
    bids == [],
    asks == [('ask', 10, 0.1, 'solar')],
    grid_transactions == (0, 0.1449, 10, 0.069),
    financial_transactions == [0, 0, 0, 0]
]
print('c7', check),
print(bids, asks, grid_transactions, financial_transactions)
print('-----')

# case 8: perfect asks for generator, charge battery
market_ledger = [('ask', 10, 0.1, 'solar')]
bids, asks, grid_transactions, financial_transactions = test_market.deliver(market_ledger, 10, 0, 10)
# expected output:
check = [
    bids == [],
    asks == [('ask', 10, 0.1, 'solar')],
    grid_transactions == (10, 0.1449, 0, 0.069),
    financial_transactions == [0, 0, 0, 0]
]
print('c8', check),
print(bids, asks, grid_transactions, financial_transactions)
print('-----')

# case 9: ask more than generation for generator, discharge
market_ledger = [('ask', 20, 0.1, 'solar')]
bids, asks, grid_transactions, financial_transactions = test_market.deliver(market_ledger, 10, 0, -10)
# expected output:
check = [bids == [],
         asks == [('ask', 20, 0.1, 'solar')],
         grid_transactions == (0, 0.1449, 0, 0.069),
         financial_transactions == [0, 0, 0, 0]]
print('c9', check)
print(bids, asks, grid_transactions, financial_transactions)
print('-----')

# case 10: ask more than generation for generator, charge
market_ledger = [('ask', 20, 0.1, 'solar')]
bids, asks, grid_transactions, financial_transactions = test_market.deliver(market_ledger, 10, 0, 10)
# expected output:
check = [bids == [],
         asks == [('ask', 10, 0.1, 'solar')],
         grid_transactions == (10, 0.1449, 0, 0.069),
         financial_transactions == [10, 1.449, 0, 0]]
print('c10', check)
print(bids, asks, grid_transactions, financial_transactions)
print('-----')

# case 11: ask less than generation for generator, discharge
market_ledger = [('ask', 5, 0.1, 'solar')]
bids, asks, grid_transactions, financial_transactions = test_market.deliver(market_ledger, 10, 0, -10)
# expected output:
check = [bids == [],
         asks == [('ask', 5, 0.1, 'solar')],
         grid_transactions == (0, 0.1449, 15, 0.069),
         financial_transactions == [0, 0, 0, 0]]
print('c11', check)
print(bids, asks, grid_transactions, financial_transactions)
print('-----')

# case 12: ask less than generation for generator, charge
market_ledger = [('ask', 5, 0.1, 'solar')]
bids, asks, grid_transactions, financial_transactions = test_market.deliver(market_ledger, 10, 0, 10)
# expected output:
check = [bids == [],
         asks == [('ask', 5, 0.1, 'solar')],
         grid_transactions == (5, 0.1449, 0, 0.069),
         financial_transactions == [0, 0, 0, 0]]
print('c12', check)
print(bids, asks, grid_transactions, financial_transactions)
print('-----')


# case 13: pure consumer, perfect discharge
market_ledger = []
bids, asks, grid_transactions, financial_transactions = test_market.deliver(market_ledger, 0, 17, -17)
# expected output:
check = [bids == [],
         asks == [],
         grid_transactions == (0, 0.1449, 0, 0.069),
         financial_transactions == [0, 0, 0, 0]]
print('c13', check)
print(bids, asks, grid_transactions, financial_transactions)
print('-----')

# case 14: pure generator, perfect charge
market_ledger = []
bids, asks, grid_transactions, financial_transactions = test_market.deliver(market_ledger, 17, 0, 17)
# expected output:
check = [bids == [],
         asks == [],
         grid_transactions == (0, 0.1449, 0, 0.069),
         financial_transactions == [0, 0, 0, 0]]
print('c14', check)
print(bids, asks, grid_transactions, financial_transactions)
print('-----')

# case 12: ask less than generation for generator, charge
market_ledger = []
bids, asks, grid_transactions, financial_transactions = test_market.deliver(market_ledger, 0, 16, -17)
# expected output:
check = [bids == [],
         asks == [],
         grid_transactions == (0, 0.1449, 1, 0.069),
         financial_transactions == [0, 0, 0, 0]]
print('c15', check)
print(bids, asks, grid_transactions, financial_transactions)
print('-----')

# case 12: ask less than generation for generator, charge
market_ledger = []
bids, asks, grid_transactions, financial_transactions = test_market.deliver(market_ledger, 17, 0, 18)
# expected output:
check = [bids == [],
         asks == [],
         grid_transactions == (1, 0.1449, 0, 0.069),
         financial_transactions == [0, 0, 0, 0]]
print('c16', check)
print(bids, asks, grid_transactions, financial_transactions)
print('-----')

# case 12: ask less than generation for generator, charge
market_ledger = []
bids, asks, grid_transactions, financial_transactions = test_market.deliver(market_ledger, 0, 17, -16)
# expected output:
check = [bids == [],
         asks == [],
         grid_transactions == (1, 0.1449, 0, 0.069),
         financial_transactions == [0, 0, 0, 0]]
print('c17', check)
print(bids, asks, grid_transactions, financial_transactions)
print('-----')

# case 12: ask less than generation for generator, charge
market_ledger = []
bids, asks, grid_transactions, financial_transactions = test_market.deliver(market_ledger, 18, 0, 17)
# expected output:
check = [bids == [],
         asks == [],
         grid_transactions == (0, 0.1449, 1, 0.069),
         financial_transactions == [0, 0, 0, 0]]
print('c18', check)
print(bids, asks, grid_transactions, financial_transactions)
print('-----')


# case 19: perfect bids for a net consumer, charge
market_ledger = [('bid', 10, 0.1, 'solar')]
bids, asks, grid_transactions, financial_transactions = test_market.deliver(market_ledger, 10, 15, 5)
# expected output:
check = [bids == [('bid', 10, 0.1, 'solar')],
         asks == [],
         grid_transactions == (0, 0.1449, 0, 0.069),
         financial_transactions == [0, 0, 0, 0]]
print('c19', check)
print(bids, asks, grid_transactions, financial_transactions)
print('-----')

# case 20: perfect bids for a net consumer, discharge
market_ledger = [('bid', 10, 0.1, 'solar')]
bids, asks, grid_transactions, financial_transactions = test_market.deliver(market_ledger, 10, 25, -5)
# expected output:
check = [bids == [('bid', 10, 0.1, 'solar')],
         asks == [],
         grid_transactions == (0, 0.1449, 0, 0.069),
         financial_transactions == [0, 0, 0, 0]]
print('c20', check)
print(bids, asks, grid_transactions, financial_transactions)
print('-----')

# case 21: bid more than consumed for a net consumer, charge
market_ledger = [('bid', 20, 0.1, 'solar')]
bids, asks, grid_transactions, financial_transactions = test_market.deliver(market_ledger, 10, 10, 10)
# expected output:
check = [bids == [('bid', 10, 0.1, 'solar')],
         asks == [],
         grid_transactions == (0, 0.1449, 0, 0.069),
         financial_transactions == [10, 1.0, 0, 0]]
print('c21', check)
print(bids, asks, grid_transactions, financial_transactions)
print('-----')

# case 22: bid more than consumed for a net consumer, discharge
market_ledger = [('bid', 20, 0.1, 'solar')]
bids, asks, grid_transactions, financial_transactions = test_market.deliver(market_ledger, 10, 20, -5)
# expected output:
check = [bids == [('bid', 10, 0.1, 'solar')],
         asks == [],
         grid_transactions == (0, 0.1449, 5, 0.069),
         financial_transactions == [10, 1.0, 0, 0]]
print('c22', check)
print(bids, asks, grid_transactions, financial_transactions)
print('-----')

# case 23: bid less than consumed for a net consumer, charge
market_ledger = [('bid', 5, 0.1, 'solar')]
bids, asks, grid_transactions, financial_transactions = test_market.deliver(market_ledger, 10, 20, 5)
check = [bids == [('bid', 5, 0.1, 'solar')],
         asks == [],
         grid_transactions == (10, 0.1449, 0, 0.069),
         financial_transactions == [0, 0, 0, 0]]
print('c23', check)
print(bids, asks, grid_transactions, financial_transactions)
print('-----')

# case 24: bid less than consumed for a net consumer, discharge
market_ledger = [('bid', 5, 0.1, 'solar')]
bids, asks, grid_transactions, financial_transactions = test_market.deliver(market_ledger, 10, 20, -5)
check = [bids == [('bid', 5, 0.1, 'solar')],
         asks == [],
         grid_transactions == (0, 0.1449, 0, 0.069),
         financial_transactions == [0, 0, 0, 0]]
print('c24', check)
print(bids, asks, grid_transactions, financial_transactions)
print('-----')

# case 25: perfect asks for a net producer, charge
market_ledger = [('ask', 10, 0.1, 'solar')]
bids, asks, grid_transactions, financial_transactions = test_market.deliver(market_ledger, 25, 10, 5)
check = [ bids == [],
          asks == [('ask', 10, 0.1, 'solar')],
          grid_transactions == (0, 0.1449, 0, 0.069),
          financial_transactions == [0, 0, 0, 0]]
print('c10', check)
print(bids, asks, grid_transactions, financial_transactions)
print('-----')

# case 26: perfect asks for a net producer, discharge
market_ledger = [('ask', 10, 0.1, 'solar')]
bids, asks, grid_transactions, financial_transactions = test_market.deliver(market_ledger, 15, 10, -5)
check = [ bids == [],
          asks == [('ask', 10, 0.1, 'solar')],
          grid_transactions == (0, 0.1449, 0, 0.069),
          financial_transactions == [0, 0, 0, 0]]
print('c26', check)
print(bids, asks, grid_transactions, financial_transactions)
print('-----')

# case 27: asks more than generated for a net producer, charge
market_ledger = [('ask', 20, 0.1, 'solar')]
bids, asks, grid_transactions, financial_transactions = test_market.deliver(market_ledger, 20, 10, 5)
check = [bids == [],
         asks == [('ask', 20, 0.1, 'solar')],
         grid_transactions == (15, 0.1449, 0, 0.069),
         financial_transactions == [0, 0, 0, 0]]

print('c27', check)
print(bids, asks, grid_transactions, financial_transactions)
print('-----')

# case 28: asks more than generated for a net producer, discharge
market_ledger = [('ask', 20, 0.1, 'solar')]
bids, asks, grid_transactions, financial_transactions = test_market.deliver(market_ledger, 20, 10, -5)
check = [bids == [],
         asks == [('ask', 20, 0.1, 'solar')],
         grid_transactions == (5, 0.1449, 0, 0.069),
         financial_transactions == [0, 0, 0, 0]]

print('c28', check)
print(bids, asks, grid_transactions, financial_transactions)
print('-----')

# case 29: asks more than generated for a net producer, net generation = ask, charge
market_ledger = [('ask', 20, 0.1, 'solar')]
bids, asks, grid_transactions, financial_transactions = test_market.deliver(market_ledger, 40, 10, 10)
check = [bids == [],
         asks == [('ask', 20, 0.1, 'solar')],
         grid_transactions == (0, 0.1449, 0, 0.069),
         financial_transactions == [0, 0, 0, 0]
]
print('c29', check)
print(bids, asks, grid_transactions, financial_transactions)
print('-----')

# case 30: asks more than generated for a net producer, net generation = ask, discharge
market_ledger = [('ask', 20, 0.1, 'solar')]
bids, asks, grid_transactions, financial_transactions = test_market.deliver(market_ledger, 20, 10, -10)
check = [bids == [],
         asks == [('ask', 20, 0.1, 'solar')],
         grid_transactions == (0, 0.1449, 0, 0.069),
         financial_transactions == [0, 0, 0, 0]
]
print('c30', check)
print(bids, asks, grid_transactions, financial_transactions)
print('-----')

# case 31: asks more than generated for a net producer, net generation less than ask, charge
market_ledger = [('ask', 20, 0.1, 'solar')]
bids, asks, grid_transactions, financial_transactions = test_market.deliver(market_ledger, 15, 10, 5)
check = [bids == [],
         asks == [('ask', 15, 0.1, 'solar')],
         grid_transactions == (15, 0.1449, 0, 0.069),
         financial_transactions == [5, 0.7245, 0, 0]]
print('c31', check)
print(bids, asks, grid_transactions, financial_transactions)
print('-----')

# case 32: asks more than generated for a net producer, net generation less than ask, discharge
market_ledger = [('ask', 20, 0.1, 'solar')]
bids, asks, grid_transactions, financial_transactions = test_market.deliver(market_ledger, 10, 10, -5)
check = [bids == [],
         asks == [('ask', 15, 0.1, 'solar')],
         grid_transactions == (10, 0.1449, 0, 0.069),
         financial_transactions == [5, 0.7245, 0, 0]]
print('c32', check)
print(bids, asks, grid_transactions, financial_transactions)
print('-----')

# case 33: asks less than generated for a net producer, net generation = ask, charge
market_ledger = [('ask', 10, 0.1, 'solar')]
bids, asks, grid_transactions, financial_transactions = test_market.deliver(market_ledger, 20, 5, 5)
check = [bids == [],
         asks == [('ask', 10, 0.1, 'solar')],
         grid_transactions == (0, 0.1449, 0, 0.069),
         financial_transactions ==[0, 0, 0, 0]]
print('c33', check),
print(bids, asks, grid_transactions, financial_transactions)
print('-----')

# case 34: asks less than generated for a net producer, net generation = ask, discharge
market_ledger = [('ask', 10, 0.1, 'solar')]
bids, asks, grid_transactions, financial_transactions = test_market.deliver(market_ledger, 15, 10, -5)
check = [bids == [],
         asks == [('ask', 10, 0.1, 'solar')],
         grid_transactions == (0, 0.1449, 0, 0.069),
         financial_transactions ==[0, 0, 0, 0]]
print('c34', check),
print(bids, asks, grid_transactions, financial_transactions)
print('-----')

# case 35: asks less than generated for a net producer, net generation > ask, charge
market_ledger = [('ask', 10, 0.1, 'solar')]
bids, asks, grid_transactions, financial_transactions = test_market.deliver(market_ledger, 30, 5, 5)
check = [bids == [],
         asks == [('ask', 10, 0.1, 'solar')],
         grid_transactions == (0, 0.1449, 10, 0.069),
         financial_transactions == [0, 0, 0, 0]]
print('c35', check)
print(bids, asks, grid_transactions, financial_transactions)
print('-----')

# case 35: asks less than generated for a net producer, net generation > ask, discharge
market_ledger = [('ask', 10, 0.1, 'solar')]
bids, asks, grid_transactions, financial_transactions = test_market.deliver(market_ledger, 25, 10, -5)
check = [bids == [],
         asks == [('ask', 10, 0.1, 'solar')],
         grid_transactions == (0, 0.1449, 10, 0.069),
         financial_transactions == [0, 0, 0, 0]]
print('c36', check)
print(bids, asks, grid_transactions, financial_transactions)
print('-----')


# case 37: asks less than generated for a net producer, net generation < ask, charge
market_ledger = [('ask', 10, 0.1, 'solar')]
bids, asks, grid_transactions, financial_transactions = test_market.deliver(market_ledger, 15, 5, 5)
check = [bids == [],
         asks == [('ask', 10, 0.1, 'solar')],
         grid_transactions == (5, 0.1449, 0, 0.069),
         financial_transactions == [0, 0, 0, 0]]
print('c37', check),
print(bids, asks, grid_transactions, financial_transactions)
print('-----')

# case 38: asks less than generated for a net producer, net generation < ask, discharge
market_ledger = [('ask', 10, 0.1, 'solar')]
bids, asks, grid_transactions, financial_transactions = test_market.deliver(market_ledger, 10, 10, -5)
check = [bids == [],
         asks == [('ask', 10, 0.1, 'solar')],
         grid_transactions == (5, 0.1449, 0, 0.069),
         financial_transactions == [0, 0, 0, 0]]
print('c38', check),
print(bids, asks, grid_transactions, financial_transactions)
print('-----')