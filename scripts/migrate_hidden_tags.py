'''
Changes # HIDDEN to hide-input cell tags

Usage: migrate_hidden_tags.py
'''
from glob import glob
from docopt import docopt
import nbformat
import re

HIDDEN = re.compile(r'#\s+HIDDEN')
TAG = ['hide-input']


def transform_file(nb_file):
    nb = nbformat.read(nb_file, as_version=4)
    dirty = False
    for cell in nb.cells:
        if cell.cell_type != 'code': continue
        if not HIDDEN.match(cell.source): continue
        dirty = True

        new_source = re.sub(HIDDEN, '', cell.source).strip()

        cell.source = new_source
        cell.metadata.tags = TAG

    if not dirty: return
    nbformat.write(nb, nb_file)
    print(f'Wrote {nb_file}')


if __name__ == '__main__':
    arguments = docopt(__doc__, version='0.1')

    # transform_file('content/ch/14/feature_one_hot.ipynb')
    notebooks = sorted(glob('content/ch/**/*.ipynb', recursive=True))
    for nb_file in notebooks:
        transform_file(nb_file)
