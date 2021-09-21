import matplotlib.pyplot as plt
import numpy as np

# plots the log
class log_plotter():

    def __init__(self, log):
        self.log = log

    def plot_returns(self, export=False):
        num_agents = len([participant for participant in self.log])
        fig, ax = plt.subplots(num_agents, 1, sharex=True)
        ax[0].set_xlabel('Generations')

        plot_nbr = 0
        for participant in self.log:
            ax[plot_nbr].set_ylabel('Return')
            ax[plot_nbr].plot(self.log[participant]['G'], label=participant)
            plot_nbr +=1

        fig.legend()
        fig.tight_layout()
        fig.show()
        # calculattes and plots returns
        # optionally exports the plot as png for use externally
        return False

    def plot_quantities(self, export=False):
        fig, ax = plt.subplots()
        ax.set_xlabel('Generations')
        ax.set_ylabel('Settled kWh')
        for participant in self.log:
            ax.plot(self.log[participant]['quantity'], label=participant)

        fig.legend()
        fig.tight_layout()
        fig.show()
        # calculattes and plots quantities
        # optionally exports the plot as png for use externally
        return False

    def plot_prices(self, export=False):
        fig, (bid_ax, ask_ax) = plt.subplots(2, 1, sharex=True)
        bid_ax.set_xlabel('Generations')
        bid_ax.set_ylabel('Bid Prices')
        ask_ax.set_ylabel('Ask Prices')
        for participant in self.log:
            bids = self.log[participant]['avg_prices']['avg_bid_price']
            bid_ax.plot(bids, label=participant)

            asks = self.log[participant]['avg_prices']['avg_ask_price']
            ask_ax.plot(asks)

        fig.legend()
        fig.tight_layout()
        fig.show()
        return False

    def __export_plot(self, fig):
        return False