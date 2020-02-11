import matplotlib.pyplot as plt
import seaborn as sns


def plot(tickers, types, data):

    for ticker in tickers:

        for plot_type in types:

            if plot_type == "histogram":

                sns.distplot(data[ticker]["daily"]["pct_change"], kde=False, bins=50)
                plt.title(ticker)
                plt.ylabel('Frequency')
                plt.show()

            elif plot_type == "percent_returns":

                data[ticker]["daily"]["cum_return"].plot()
                plt.title(ticker)
                plt.ylabel('Returns (%)')
                plt.show()

