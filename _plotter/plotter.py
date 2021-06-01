import matplotlib.pyplot as plt
import numpy as np

# plots the log
class log_plotter():

    def __init__(self, log):
        self.log = log

    def plot_returns(self, export=False):
        num_agents = len([participant for participant in self.log])
        g_fig, g_ax = plt.subplots(num_agents, 1, sharex=True)
        g_ax[0].set_xlabel('Generations')

        plot_nbr = 0
        for participant in self.log:
            g_ax[plot_nbr].set_ylabel('Return')
            g_ax[plot_nbr].plot(self.log[participant]['G'], label=participant)
            plot_nbr +=1

        g_fig.legend()
        g_fig.show()
        # calculattes and plots returns
        # optionally exports the plot as png for use externally
        return False

    def plot_quantities(self, export=False):
        quant_fig, quant_ax = plt.subplots()
        quant_ax.set_xlabel('Generations')
        quant_ax.set_ylabel('Settled kWh')
        for participant in self.log:
            quant_ax.plot(self.log[participant]['quant'], label=participant)

        quant_fig.legend()
        quant_fig.show()
        # calculattes and plots quantities
        # optionally exports the plot as png for use externally
        return False

    def plot_prices(self, export=False):
        price_fig, (bid_ax, ask_ax) = plt.subplots(2, 1, sharex=True)
        bid_ax.set_xlabel('Generations')
        bid_ax.set_ylabel('Bid Prices')
        ask_ax.set_ylabel('Ask Prices')
        for participant in self.log:
            bids = self.log[participant]['avg_prices']['avg_bid_price']
            bid_ax.plot(bids, label=participant)

            asks = self.log[participant]['avg_prices']['avg_ask_price']
            ask_ax.plot(asks)

        price_fig.legend()
        price_fig.show()
        return False

    def __export_plot(self, fig):
        return False