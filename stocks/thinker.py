import datetime
import pandas as pd
import numpy as np

pd.set_option('display.max_rows', None)


def scrape():
    # TODO this can just scrape either yahoo, bloomberg, wsj, or marketwatch for top news on preferred companies.
    # TODO look for dips. i.e. either 1 std or 0.5 std

    pass


def thinker(ticker, date, stock_data, options_data, scenario, *args):
    """ get insight into specific option.
    ticker = str of ticker
    date = str of date (YYYY-MM-DD)
    scenario = nominal, bullish, bearish
    insight_date = date when news anticipated to affect stock price
    duration = how long it will have an effect, multiple of weeks
    """
    if args:
        insight_date = args[0]
        duration = args[1]

    today = datetime.date.today()
    end_date = datetime.datetime.strptime(date, "%Y-%m-%d").date()
    latest_price = stock_data[ticker]['end_price']
    weekly_std = stock_data[ticker]["weekly"]['std']
    daily_mu = stock_data[ticker]["daily"]['mean']
    daily_std = stock_data[ticker]["daily"]['std']

    # TODO determine which distribution fits data best? then determine probability of price point.

    if scenario == 'nominal':
        # all days
        #dt = int((end_date - today).days)
        # business days only
        dt = np.busday_count(today, datetime.datetime.strptime(date, "%Y-%m-%d").date())
        # conservative outlook, use 75% of avg and BD only.
        mu = daily_mu * 0.75
        #mu = daily_mu
        x_f = latest_price * (1 + mu) ** dt

    # bullish news
    elif scenario == 'bullish':

        try:

            # calc price up until insight date.
            insight_date = datetime.datetime.strptime(insight_date, "%Y-%m-%d").date()
            # dt_1 = int((insight_date - today).days)
            dt_1 = np.busday_count(today, insight_date)
            mu = 0.75 * daily_mu
            x_1 = latest_price * (1 + mu) ** dt_1
            x_2 = x_1 * (1 + weekly_std) ** duration

            # dt_2 = int((end_date - (insight_date + datetime.timedelta(weeks=duration))).days)
            dt_2 = np.busday_count((insight_date + datetime.timedelta(weeks=duration)), end_date)
            x_f = x_2 * (1 + mu) ** dt_2


        except:

            print("Error. Make sure correct arguments entered.")

        # x_1 = stock_data[ticker]['end_price'] * (1 + weekly_std)
        # dt = int((datetime.datetime.strptime(date, "%Y-%m-%d").date() - (today + datetime.timedelta(days=7))).days)
        # x_f =  x_1 * (1 + daily_mu)**dt

    # bearish news.
    elif scenario == 'bearish':

        pass

    else:

        print("Error. This capability has not been added yet.")

    total_change = (x_f - latest_price) / latest_price * 100
    print(f"scenario = {scenario}")
    print('current price:', latest_price)
    print(f'predicted price on {date}:', x_f)
    print(f'total % change: {total_change}')
    print(f'weekly_std = {weekly_std} | daily_mu = {daily_mu} | daily_std = {daily_std}')
    print(f'expiration date is {date}')
    print(options_data[date][['strike', 'lastPrice', 'volume', 'impliedVolatility']])

    return None
