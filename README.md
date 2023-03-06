# tendies-maker

Make tendies

## Installation

Packages required are included in requirements.txt
Use the package manager [pip](https://pip.pypa.io/en/stable/) to 
install, or any desired package manager.

```bash
pip install -r requirements.txt
```

## Usage
Make sure to set desired data directory in header of files.
### Gathering data
TrainingData object is used to gather data from various sources. 
* **wsb** - Scrapes r/wallstreetbets via https://stocks.comment.ai/ to track mentions of 
tickers and sentiment.
* **shorts** - Scrapes https://www.highshortinterest.com/ to find most shorted stocks
* **indicators**  - technical indicator values for wsb tickers
   * [MACD](https://www.investopedia.com/terms/m/macd.asp)
   * [Ichimoku clouds](https://www.investopedia.com/terms/i/ichimoku-cloud.asp#:~:text=The%20Ichimoku%20Cloud%20is%20a,plotting%20them%20on%20the%20chart.)
   * [RSI](https://www.investopedia.com/terms/r/rsi.asp)
   * [%B](https://school.stockcharts.com/doku.php?id=technical_indicators:bollinger_band_perce) 
* **news** - Scrapes news headlines from https://finviz.com/ and runs natural language sentiment analysis to generate score
* **earnings** - Uses yfinance to get days until next earnings event
* **changes** - pulls data on largest change in past month and how many days since then
* **options** - pulls options data and calculates put-call ratio (volume) and then value ratio of puts to calls

### portfolio.py
This has code inside to optimize for maximum Sharpe Ratio given a list of tickers. Also has some commented
code useful for getting options chain for tickers. This file is less user friendly and cannot be run with arguments
### ML
TBA