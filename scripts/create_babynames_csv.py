'''
Creates babynames.csv file by concatenating baby names split by year. Needs
a folder in the same directory called `names/` with the text files in it,
created by downloading and extracting the National data zipfile from:
https://www.ssa.gov/oact/babynames/limits.html

Usage: create_babynames_csv.py
'''

import pandas as pd
from pathlib import Path

year_files = list(Path('names').glob('*.txt'))


def read_year(path):
    year = int(path.stem[3:])
    return (pd.read_csv(path, header=0, names=['Name', 'Sex',
                                               'Count']).assign(Year=year))


baby = (pd.concat(map(read_year,
                      year_files)).sort_values(['Year', 'Sex', 'Count'],
                                               ascending=False))

baby.to_csv('babynames.csv', index=False)
