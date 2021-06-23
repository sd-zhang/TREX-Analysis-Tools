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

# case 1: perfect bids for consumer
market_ledger = [('bid', 10, 0.1, 'solar')]
bids, asks, grid_transactions, financial_transactions = test_market.deliver(market_ledger, 0, 10)

check = [bids == [('bid', 10, 0.1, 'solar')],
         asks == [],
         grid_transactions == (0, 0.1449, 0, 0.069),
         financial_transactions == [0, 0, 0, 0]]
print('c1', check)
print(bids, asks, grid_transactions, financial_transactions)
print('-----')

# case 2: bid more than consumption for consumer
market_ledger = [('bid', 20, 0.1, 'solar')]
bids, asks, grid_transactions, financial_transactions = test_market.deliver(market_ledger, 0, 10)

check = [bids == [('bid', 10, 0.1, 'solar')],
         asks == [],
         grid_transactions == (0, 0.1449, 0, 0.069),
         financial_transactions == [10, 1.0, 0, 0]]
print('c2', check)
print(bids, asks, grid_transactions, financial_transactions)
print('-----')

# case 3: bid less than consumption for consumer
market_ledger = [('bid', 5, 0.1, 'solar')]
bids, asks, grid_transactions, financial_transactions = test_market.deliver(market_ledger, 0, 10)
# expected output:
check = [bids == [('bid', 5, 0.1, 'solar')],
         asks == [],
         grid_transactions == (5, 0.1449, 0, 0.069),
         financial_transactions == [0, 0, 0, 0]]
print('c3', check)
print(bids, asks, grid_transactions, financial_transactions)
print('-----')

# case 4: perfect asks for generator
market_ledger = [('ask', 10, 0.1, 'solar')]
bids, asks, grid_transactions, financial_transactions = test_market.deliver(market_ledger, 10, 0)
# expected output:
check = [
    bids == [],
    asks == [('ask', 10, 0.1, 'solar')],
    grid_transactions == (0, 0.1449, 0, 0.069),
    financial_transactions == [0, 0, 0, 0]
]
print('c4', check),
print(bids, asks, grid_transactions, financial_transactions)
print('-----')

# case 5: ask more than generation for generator
market_ledger = [('ask', 20, 0.1, 'solar')]
bids, asks, grid_transactions, financial_transactions = test_market.deliver(market_ledger, 10, 0)
# expected output:
check = [bids == [],
         asks == [('ask', 10, 0.1, 'solar')],
         grid_transactions == (0, 0.1449, 0, 0.069),
         financial_transactions == [10, 1.4490, 0, 0]]
print('c5', check)
print(bids, asks, grid_transactions, financial_transactions)
print('-----')
# case 6: ask less than generation for generator
market_ledger = [('ask', 5, 0.1, 'solar')]
bids, asks, grid_transactions, financial_transactions = test_market.deliver(market_ledger, 10, 0)
# expected output:
check = [bids == [],
         asks == [('ask', 5, 0.1, 'solar')],
         grid_transactions == (0, 0.1449, 5, 0.069),
         financial_transactions == [0, 0, 0, 0]]
print('c6', check)
print(bids, asks, grid_transactions, financial_transactions)
print('-----')

# case 7: perfect bids for a net consumer
market_ledger = [('bid', 10, 0.1, 'solar')]
bids, asks, grid_transactions, financial_transactions = test_market.deliver(market_ledger, 10, 20)
# expected output:
check = [bids == [('bid', 10, 0.1, 'solar')],
         asks == [],
         grid_transactions == (0, 0.1449, 0, 0.069),
         financial_transactions == [0, 0, 0, 0]]
print('c7', check)
print(bids, asks, grid_transactions, financial_transactions)
print('-----')

# case 8: bid more than consumed for a net consumer
market_ledger = [('bid', 20, 0.1, 'solar')]
bids, asks, grid_transactions, financial_transactions = test_market.deliver(market_ledger, 10, 20)
# expected output:
check = [bids == [('bid', 10, 0.1, 'solar')],
         asks == [],
         grid_transactions == (0, 0.1449, 0, 0.069),
         financial_transactions == [10, 1.0, 0, 0]]
print('c8', check)
print(bids, asks, grid_transactions, financial_transactions)
print('-----')

# case 9: bid less than consumed for a net consumer
market_ledger = [('bid', 5, 0.1, 'solar')]
bids, asks, grid_transactions, financial_transactions = test_market.deliver(market_ledger, 10, 20)
check = [bids == [('bid', 5, 0.1, 'solar')],
         asks == [],
         grid_transactions == (5, 0.1449, 0, 0.069),
         financial_transactions == [0, 0, 0, 0]]
print('c9', check)
print(bids, asks, grid_transactions, financial_transactions)
print('-----')

# case 10: perfect asks for a net producer
market_ledger = [('ask', 10, 0.1, 'solar')]
bids, asks, grid_transactions, financial_transactions = test_market.deliver(market_ledger, 20, 10)
check = [ bids == [],
          asks == [('ask', 10, 0.1, 'solar')],
          grid_transactions == (0, 0.1449, 0, 0.069),
          financial_transactions == [0, 0, 0, 0]]
print('c10', check)
print(bids, asks, grid_transactions, financial_transactions)
print('-----')

# case 11: asks more than generated for a net producer
market_ledger = [('ask', 20, 0.1, 'solar')]
bids, asks, grid_transactions, financial_transactions = test_market.deliver(market_ledger, 20, 10)
check = [bids == [],
         asks == [('ask', 20, 0.1, 'solar')],
         grid_transactions == (10, 0.1449, 0, 0.069),
         financial_transactions == [0, 0, 0, 0]]

print('c11', check)
print(bids, asks, grid_transactions, financial_transactions)
print('-----')
# expected output:
# [] [('ask', 20, 0.1, 'solar')] (10, 0.1449, 0, 0.069) [0, 0, 0, 0]

# case 12: asks more than generated for a net producer, net generation = ask
market_ledger = [('ask', 20, 0.1, 'solar')]
bids, asks, grid_transactions, financial_transactions = test_market.deliver(market_ledger, 30, 10)
check = [bids == [],
         asks == [('ask', 20, 0.1, 'solar')],
         grid_transactions == (0, 0.1449, 0, 0.069),
         financial_transactions == [0, 0, 0, 0]
]
print('c12', check)
print(bids, asks, grid_transactions, financial_transactions)
print('-----')

# case 13: asks more than generated for a net producer, net generation less than ask
market_ledger = [('ask', 20, 0.1, 'solar')]
bids, asks, grid_transactions, financial_transactions = test_market.deliver(market_ledger, 15, 10)
check = [bids == [],
         asks == [('ask', 15, 0.1, 'solar')],
         grid_transactions == (10, 0.1449, 0, 0.069),
         financial_transactions == [5, 0.7245, 0, 0]]
print('c13', check)
print(bids, asks, grid_transactions, financial_transactions)
print('-----')
# expected output:
# [] [('ask', 15, 0.1, 'solar')] (10, 0.1449, 0, 0.069) [5, 0.7245, 0, 0]

# case 14: asks less than generated for a net producer, net generation = ask
market_ledger = [('ask', 10, 0.1, 'solar')]
bids, asks, grid_transactions, financial_transactions = test_market.deliver(market_ledger, 20, 10)
check = [bids == [],
         asks == [('ask', 10, 0.1, 'solar')],
         grid_transactions == (0, 0.1449, 0, 0.069),
         financial_transactions ==[0, 0, 0, 0]]
print('c14', check),
print(bids, asks, grid_transactions, financial_transactions)
print('-----')
# expected output:
# [] [('ask', 10, 0.1, 'solar')] (0, 0.1449, 0, 0.069) [0, 0, 0, 0]

# case 15: asks less than generated for a net producer, net generation > ask
market_ledger = [('ask', 10, 0.1, 'solar')]
bids, asks, grid_transactions, financial_transactions = test_market.deliver(market_ledger, 30, 10)
check = [bids == [],
         asks == [('ask', 10, 0.1, 'solar')],
         grid_transactions == (0, 0.1449, 10, 0.069),
         financial_transactions == [0, 0, 0, 0]]
print('c15', check)
print(bids, asks, grid_transactions, financial_transactions)
print('-----')
# expected output:
# [] [('ask', 10, 0.1, 'solar')] (0, 0.14490000000000003, 10, 0.069) [0, 0, 0, 0]

# case 16: asks less than generated for a net producer, net generation < ask
market_ledger = [('ask', 10, 0.1, 'solar')]
bids, asks, grid_transactions, financial_transactions = test_market.deliver(market_ledger, 15, 10)
check = [bids == [],
         asks == [('ask', 10, 0.1, 'solar')],
         grid_transactions == (5, 0.1449, 0, 0.069),
         financial_transactions == [0, 0, 0, 0]]
print('c16', check),
print(bids, asks, grid_transactions, financial_transactions)
print('-----')
# expected output:
# [] [('ask', 10, 0.1, 'solar')] (0, 0.14490000000000003, 10, 0.069) [0, 0, 0, 0]

# case 16: asks less than generated for a net producer, net generation < ask
market_ledger = []
bids, asks, grid_transactions, financial_transactions = test_market.deliver(market_ledger, 10, 1)
check = [bids == [],
         asks == [],
         grid_transactions == (0, 0.1449, 9, 0.069),
         financial_transactions == [0, 0, 0, 0]]
print('c17', check),
print(bids, asks, grid_transactions, financial_transactions)
print('-----')
# expected output:
# [] [('ask', 10, 0.1, 'solar')] (0, 0.14490000000000003, 10, 0.069) [0, 0, 0, 0]

# case 16: asks less than generated for a net producer, net generation < ask
market_ledger = []
bids, asks, grid_transactions, financial_transactions = test_market.deliver(market_ledger, 1, 10)
check = [bids == [],
         asks == [],
         grid_transactions == (9, 0.1449, 0, 0.069),
         financial_transactions == [0, 0, 0, 0]]
print('c18', check),
print(bids, asks, grid_transactions, financial_transactions)
print('-----')
# expected output:
# [] [('ask', 10, 0.1, 'solar')] (0, 0.14490000000000003, 10, 0.069) [0, 0, 0, 0]

# case 16: asks less than generated for a net producer, net generation < ask
market_ledger = []
bids, asks, grid_transactions, financial_transactions = test_market.deliver(market_ledger, 0, 0)
check = [bids == [],
         asks == [],
         grid_transactions == (0, 0.1449, 0, 0.069),
         financial_transactions == [0, 0, 0, 0]]
print('c19', check),
print(bids, asks, grid_transactions, financial_transactions)
print('-----')
# expected output:
# [] [('ask', 10, 0.1, 'solar')] (0, 0.14490000000000003, 10, 0.069) [0, 0, 0, 0]
