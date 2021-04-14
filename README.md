# tendies-maker

Make tendies

## Installation

Packages required are included in requirements.txt
Use the package manager [pip](https://pip.pypa.io/en/stable/) to 
install, or any desired package manager.

```bash
pip install -r requirements.txt
```

If using conda,
```bash
conda install --file requirements.txt
```

## Usage
### Gathering data
Run within virtual environment, conda environment, or using any Python installation with 
required libraries. Get commands help using -h
```python
python generate_data.py -h
```

Sample command to generate all data
```bash
python generate_data.py -wsb -shorts -indicators -news -earnings -changes -options
```
Use overwrite argument to not be prompted during each edit to data file and overwrite all
```bash
-overwrite
```
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

### Preprocessing
#### Results
Getting results after market close, will essentially return a 0 or 1 if stock was a buy that day. Do this before normalizing data.
```python
python prepare_data.py -results
```
#### Normalization
Two options available
- [min_max](https://en.wikipedia.org/wiki/Feature_scaling)
- [standard](https://en.wikipedia.org/wiki/Standard_score)
```python
python prepare_data.py -normalize min_max
```
## Contributing
Pull as needed, branch the code if making any changes. Also try to keep track of tasks with issues.

## License
[MIT](https://choosealicense.com/licenses/mit/)

## Requirements

- [pandas](https://pandas.pydata.org/) - Python data analysis library
- [numpy](https://numpy.org/doc/stable/) - Scientific computing package
- [vader nltk](https://www.nltk.org/_modules/nltk/sentiment/vader.html) - Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for
Sentiment Analysis of Social Media Text. Eighth International Conference on
Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June 2014.