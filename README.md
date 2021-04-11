# tendies-maker

Make tendies

## Installation

Packages required are included in requirements.txt
Use the package manager [pip](https://pip.pypa.io/en/stable/) to 
install, or any desired package manager.

```bash
pip install -r -requirements.txt
```

If using conda,
```bash
conda install --file requirements.txt
```

## Usage

Run within virtual environment, conda environment, or using any Python installation with 
required libraries. Get commands using -h
```python
python generate_data.py -h
```

Sample command to generate all data
```bash
python generate_data.py -wsb -shorts -indicators -news -earnings -changes -options
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
