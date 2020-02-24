import matplotlib.pyplot as plt
import seaborn as sns


def plot(tickers, types, timespan, data):

    for ticker in tickers:

        for plot_type in types:

            if plot_type == "histogram":

                sns.distplot(data[ticker][timespan]["pct_change"], kde=True, bins=100)
                plt.title(ticker)
                plt.ylabel('Frequency')
                plt.xlabel(f'{timespan} change')
                plt.show()

            elif plot_type == "percent_returns":

                data[ticker]["daily"]["cum_return"].plot()
                plt.title(ticker)
                plt.ylabel('Returns (%)')
                plt.show()

