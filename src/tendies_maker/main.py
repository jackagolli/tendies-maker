import utils
import pandas as pd

api_key = "BXLa1HFQP9nL_lXIPKapw7fKDTD3f_WU"


#response_test = utils.create_mock_raw_data(api_key, "SPY", "minute", 5, '2023-01-09', '2023-01-12')

#response_df = utils.parse_mock_raw_dat(response_test.json()["results"], True,
#                                       'D:/PythonProject/tm/data/mock_data.parquet')

test = pd.read_parquet('D:/PythonProject/tm/data/mock_data.parquet')

debugger = 1