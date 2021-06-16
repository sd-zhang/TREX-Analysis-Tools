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

# # case 1: perfect bids for consumer
market_ledger = [('bid', 10, 0.1, 'solar')]
bids, asks, grid_transactions, financial_transactions = test_market.deliver(market_ledger, 0, 10)
# expected output:
# expected = "[('bid', 10, 0.1, 'solar')] [] (0, 0.14490000000000003, 0, 0.069) [0, 0, 0, 0]"
print('c1', bids, asks, grid_transactions, financial_transactions)
print('-----')

# case 2: bid more than consumption for consumer
market_ledger = [('bid', 20, 0.1, 'solar')]
bids, asks, grid_transactions, financial_transactions = test_market.deliver(market_ledger, 0, 10)
# expected output:
# [('bid', 10, 0.1, 'solar')] [] (0, 0.14490000000000003, 0, 0.069) [10, 1.0, 0, 0]
print('c2', bids, asks, grid_transactions, financial_transactions)
print('-----')

# case 3: bid less than consumption for consumer
market_ledger = [('bid', 5, 0.1, 'solar')]
bids, asks, grid_transactions, financial_transactions = test_market.deliver(market_ledger, 0, 10)
# expected output:
# [('bid', 5, 0.1, 'solar')] [] (5, 0.14490000000000003, 0, 0.069) [0, 0, 0, 0]
print('c3', bids, asks, grid_transactions, financial_transactions)
print('-----')

# case 4: perfect asks for generator
market_ledger = [('ask', 10, 0.1, 'solar')]
bids, asks, grid_transactions, financial_transactions = test_market.deliver(market_ledger, 10, 0)
# expected output:
# [] [('ask', 10, 0.1, 'solar')] (0, 0.14490000000000003, 0, 0.069) [0, 0, 0, 0]
print('c4', bids, asks, grid_transactions, financial_transactions)
print('-----')

# case 5: ask more than generation for generator
market_ledger = [('ask', 20, 0.1, 'solar')]
bids, asks, grid_transactions, financial_transactions = test_market.deliver(market_ledger, 10, 0)
# expected output:
# [] [('ask', 10, 0.1, 'solar')] (0, 0.14490000000000003, 0, 0.069) [10, 1.4490000000000003, 0, 0]
print('c5', bids, asks, grid_transactions, financial_transactions)
print('-----')
# case 6: ask less than generation for generator
market_ledger = [('ask', 5, 0.1, 'solar')]
bids, asks, grid_transactions, financial_transactions = test_market.deliver(market_ledger, 10, 0)
# expected output:
# [] [('ask', 5, 0.1, 'solar')] (0, 0.14490000000000003, 5, 0.069) [0, 0, 0, 0]
print('c6', bids, asks, grid_transactions, financial_transactions)
print('-----')

# case 7: perfect bids for a net consumer
market_ledger = [('bid', 10, 0.1, 'solar')]
bids, asks, grid_transactions, financial_transactions = test_market.deliver(market_ledger, 10, 20)
# expected output:
# [('bid', 10, 0.1, 'solar')] [] (0, 0.14490000000000003, 0, 0.069) [0, 0, 0, 0]
print('c7', bids, asks, grid_transactions, financial_transactions)
print('-----')

# case 8: bid more than consumed for a net consumer
market_ledger = [('bid', 20, 0.1, 'solar')]
bids, asks, grid_transactions, financial_transactions = test_market.deliver(market_ledger, 10, 20)
# expected output:
# [('bid', 10, 0.1, 'solar')] [] (0, 0.14490000000000003, 0, 0.069) [10, 1.0, 0, 0]
print('c8', bids, asks, grid_transactions, financial_transactions)
print('-----')

# case 9: bid less than consumed for a net consumer
market_ledger = [('bid', 5, 0.1, 'solar')]
bids, asks, grid_transactions, financial_transactions = test_market.deliver(market_ledger, 10, 20)
print('c9', bids, asks, grid_transactions, financial_transactions)
print('-----')
# expected output:
# [('bid', 5, 0.1, 'solar')] [] (5, 0.14490000000000003, 0, 0.069) [0, 1.0, 0, 0]

# case 10: perfect asks for a net producer
market_ledger = [('ask', 10, 0.1, 'solar')]
bids, asks, grid_transactions, financial_transactions = test_market.deliver(market_ledger, 20, 10)
print('c10', bids, asks, grid_transactions, financial_transactions)
print('-----')
# expected output:
# [] [('ask', 10, 0.1, 'solar')] (0, 0.14490000000000003, 0, 0.069) [0, 0, 0, 0]

# case 11: asks more than generated for a net producer
market_ledger = [('ask', 20, 0.1, 'solar')]
bids, asks, grid_transactions, financial_transactions = test_market.deliver(market_ledger, 20, 10)
print('c11', bids, asks, grid_transactions, financial_transactions)
print('-----')
# expected output:
# [] [('ask', 20, 0.1, 'solar')] (10, 0.14490000000000003, 0, 0.069) [0, 0, 0, 0]

# case 12: asks more than generated for a net producer, net generation = ask
market_ledger = [('ask', 20, 0.1, 'solar')]
bids, asks, grid_transactions, financial_transactions = test_market.deliver(market_ledger, 30, 10)
print('c12', bids, asks, grid_transactions, financial_transactions)
print('-----')
# expected output:
# [] [('ask', 20, 0.1, 'solar')] (0, 0.14490000000000003, 0, 0.069) [0, 0, 0, 0]

# case 13: asks more than generated for a net producer, net generation less than ask
market_ledger = [('ask', 20, 0.1, 'solar')]
bids, asks, grid_transactions, financial_transactions = test_market.deliver(market_ledger, 15, 10)
print('c13', bids, asks, grid_transactions, financial_transactions)
print('-----')
# expected output:
# [] [('ask', 15, 0.1, 'solar')] (0, 0.14490000000000003, 0, 0.069) [5, 0.7245000000000001, 0, 0]

# case 14: asks less than generated for a net producer, net generation = ask
market_ledger = [('ask', 10, 0.1, 'solar')]
bids, asks, grid_transactions, financial_transactions = test_market.deliver(market_ledger, 20, 10)
print('c14', bids, asks, grid_transactions, financial_transactions)
print('-----')
# expected output:
# [] [('ask', 10, 0.1, 'solar')] (0, 0.14490000000000003, 0, 0.069) [0, 0, 0, 0]

# case 15: asks less than generated for a net producer, net generation > ask
market_ledger = [('ask', 10, 0.1, 'solar')]
bids, asks, grid_transactions, financial_transactions = test_market.deliver(market_ledger, 30, 10)
print('c15', bids, asks, grid_transactions, financial_transactions)
print('-----')
# expected output:
# [] [('ask', 10, 0.1, 'solar')] (0, 0.14490000000000003, 10, 0.069) [0, 0, 0, 0]

# case 16: asks less than generated for a net producer, net generation < ask
market_ledger = [('ask', 10, 0.1, 'solar')]
bids, asks, grid_transactions, financial_transactions = test_market.deliver(market_ledger, 15, 10)
print(bids, asks, grid_transactions, financial_transactions)
print('-----')
# expected output:
# [] [('ask', 10, 0.1, 'solar')] (0, 0.14490000000000003, 10, 0.069) [0, 0, 0, 0]