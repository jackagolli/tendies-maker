import utils
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv('POLYGON_API_KEY')

#response_test = utils.create_mock_raw_data(api_key, "SPY", "minute", 5, '2023-01-09', '2023-01-12')

#response_df = utils.parse_mock_raw_dat(response_test.json()["results"], False, True,
#                                       'D:/PythonProject/tm/data/mock_data.parquet')

test = pd.read_parquet('D:/PythonProject/tm/data/mock_data.parquet')

debugger = 1