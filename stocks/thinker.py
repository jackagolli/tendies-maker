import datetime
import pandas as pd
pd.set_option('display.max_rows', None)

def scrape():

    #TODO this can just scrape either yahoo, bloomberg, wsj, or marketwatch for top news on preferred companies.
    #TODO look for dips. i.e. either 1 std or 0.5 std

    pass


def thinker(ticker, date, stock_data, options_data, scenario, *args):

    if args:
        insight_date = args[0]
        duration = args[1]

    today = datetime.date.today()
    weekly_std = stock_data[ticker]["weekly"]['std']
    daily_mu = stock_data[ticker]["daily"]['mean']
    daily_std = stock_data[ticker]["daily"]['std']

    # TODO determine which distribution fits data best? then determine probability of price point.

    # Need high implied volatility. Need to print out standard deviation and the mean.
    if scenario == 'nominal':

        dt = int((datetime.datetime.strptime(date, "%Y-%m-%d").date() - today).days)
        x_f = stock_data[ticker]['end_price'] * (1 + daily_mu )**dt

    # bullish news
    elif scenario == 'bullish':

        try:

            # calc price up until insight date.
            insight_date = datetime.datetime.strptime(insight_date, "%Y-%m-%d").date()
            dt_1 = int((insight_date - today).days)
            x_1 = stock_data[ticker]['end_price'] * (1 + daily_mu)**dt_1
            x_2 = x_1 * (1 + weekly_std)**duration

            dt_2 = int((datetime.datetime.strptime(date, "%Y-%m-%d").date() - (insight_date + datetime.timedelta(weeks=duration))).days)
            x_f = x_2 * (1 + daily_mu) ** dt_2


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

    print(f"scenario = {scenario}")
    print('current price:', stock_data[ticker]['end_price'])
    print(f'predicted price on {date}:', x_f)
    print(f'weekly_std = {weekly_std}, daily_mu = {daily_mu}, daily_std = {daily_std}')
    print(f'expiration date is {date}')
    print(options_data[date][['strike', 'lastPrice']])

    return None
