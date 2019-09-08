"""Usage: add_hidden_tags.py [NOTEBOOKS ...]

If a cell starts with `# HIDDEN`, add a `hide_input` tag to the cell so that
the cell stays hidden when using jupyter-book.

Arguments:
    NOTEBOOKS  Notebook files to convert. By default, will convert all
               notebooks in the notebooks/ folder.
"""

import re
import nbformat
import glob
from docopt import docopt

HIDE_TAG = 'hide_input'

has_hidden = re.compile(r'#\s?HIDDEN').search


def mapl(fn, iterable):
    return list(map(fn, iterable))


def read_notebook(nb_path):
    return nbformat.read(nb_path, as_version=4)


def write_notebook(nb, nb_path):
    return nbformat.write(nb, nb_path)


def add_tag_if_needed(cell):
    if not has_hidden(cell.source):
        return cell

    tags = cell.metadata.get('tags', [])
    if HIDE_TAG not in tags:
        tags.append(HIDE_TAG)
    cell.metadata.tags = tags
    return cell


if __name__ == '__main__':
    arguments = docopt(__doc__)
    notebooks = arguments['NOTEBOOKS'] or sorted(
        glob.glob('notebooks/**/*.ipynb', recursive=True))

    for nb_path in notebooks:
        nb = read_notebook(nb_path)

        nb.cells = mapl(add_tag_if_needed, nb.cells)
        write_notebook(nb, nb_path)
        print('Wrote {}'.format(nb_path))
