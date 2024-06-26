import stocks
from pathlib import Path
import datetime as dt
import multiprocessing as mp
import pandas as pd
import os
import re
import argparse

# Set data directory here
data_dir = Path.cwd() / 'data' / 'ml'
if data_dir.exists():
    pass
else:
    Path.mkdir(data_dir)

today = dt.date.today().strftime("%m-%d-%Y")
num_proc = mp.cpu_count() - 1


parser = argparse.ArgumentParser()
parser.add_argument("-results", help="generates results column for given day",
                    action="store_true")
parser.add_argument("-normalize", help="normalizes data, specify either 'min_max' or 'standard'",
                )
args = parser.parse_args()


if args.normalize:
    """
        
        Normalize data.
    
    """
    param = args.normalize
    if param == 'min_max' or param == 'standard':

        files = []

        for file in os.listdir(data_dir):

            match = re.search(r'\d\d-\d\d-\d\d\d\d', file)

            if match and not any(x in file for x in ['sentiment', 'normalized']):
                files.append(file)
                # df = pd.read_csv(data_dir / file, index_col=0, dtype={"mentions": np.int32})

        files = sorted(files, key=lambda x: dt.datetime.strptime(
            re.search(r'\d\d-\d\d-\d\d\d\d', x)[0], "%m-%d-%Y"), reverse=True)

        for file in files:
            path = data_dir / file
            date_str = re.search(r'\d\d-\d\d-\d\d\d\d', file)[0]
            data = pd.read_csv(path, index_col=0)

            if param == "min_max":

                try:

                    normalized_data = stocks.min_max_normalize(data, ignored_columns=['put_call_ratio',
                                                                                      'put_call_value_ratio',
                                                                                      'max_intraday_change_1mo',
                                                                                      'sentiment',
                                                                                      'short_interest','Y'])
                except:

                    print("Check if 'Y' column is present.")
                    quit()

            elif param == "standard":

                normalized_data = stocks.standard_score_normalize(data, ignored_columns=['Y'])

            if Path(data_dir / ("normalized_data_" + date_str + ".csv")).is_file():
                overwrite = input(f'Normalized data file for {date_str} already exists for today. Overwrite (Y/N)? ')

                if overwrite == 'Y' or overwrite == 'y':
                    normalized_data.to_csv(data_dir / ("normalized_data_" + date_str + ".csv"))
                elif overwrite == 'N' or overwrite == 'n':
                    pass
                else:
                    print('Invalid input.')
            else:
                normalized_data.to_csv(data_dir / ("normalized_data_" + date_str + ".csv"))



    else:
        print("Please specify one of two required normalization types.")

if args.results:
    """
    Get results for the day. Run after market close.
    """
    files = []

    for file in os.listdir(data_dir):

        match = re.search(r'\d\d-\d\d-\d\d\d\d', file)

        if match and not any(x in file for x in ['sentiment', 'normalized']):
            files.append(file)
            # df = pd.read_csv(data_dir / file, index_col=0, dtype={"mentions": np.int32})

    files = sorted(files, key=lambda x: dt.datetime.strptime(
        re.search(r'\d\d-\d\d-\d\d\d\d', x)[0], "%m-%d-%Y"), reverse=True)

    for file in files:
        path = data_dir / file
        date_str = re.search(r'\d\d-\d\d-\d\d\d\d', file)[0]
        date2 = dt.datetime.strptime(date_str,"%m-%d-%Y")
        # Check if there was spike in past 2 days for target column
        date1 = date2 + dt.timedelta(days=1)
        date3 = date2 - dt.timedelta(days=1)

        weekno = date1.weekday()
        data = pd.read_csv(path, index_col=0)
        tickers = stocks.gather_wsb_tickers(data_dir, date_str)

        bd = date3.strftime("%Y-%m-%d")
        sd = date2.strftime("%Y-%m-%d")
        ed = date1.strftime("%Y-%m-%d")

        if weekno == 5 or weekno == 6:
            prices = stocks.gather_multi(tickers, period="1d", key='all')
        else:
            prices = stocks.gather_multi(tickers, start_date=bd, end_date=ed, key='all')

        result = stocks.gather_results(prices,tickers)
        stocks.append_to_table(data_dir,data=result,date_str=date_str,overwrite=False)

