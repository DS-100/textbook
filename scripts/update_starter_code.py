"""Usage: update_starter_code.py [NOTEBOOKS ...]

Updates every notebook's first cell to match the first cell of starter.ipynb.

Used basically whenever the code in starter.ipynb changes.

Arguments:
    NOTEBOOKS  Notebook files to convert. By default, will convert all
               notebooks in the notebooks/ folder.
"""

import nbformat
import glob
from docopt import docopt

STARTER_NB = 'starter.ipynb'


def read_notebook(nb_path):
    return nbformat.read(nb_path, as_version=4)


def write_notebook(nb, nb_path):
    return nbformat.write(nb, nb_path)


def get_starter_cell(starter_nb=STARTER_NB):
    nb = read_notebook(starter_nb)
    return nb.cells[0]


def update_first_cell(nb, starter_cell):
    nb.cells[0] = starter_cell
    return nb


def should_update_nb(nb):
    return nb.cells[0].cell_type == 'code'


if __name__ == '__main__':
    arguments = docopt(__doc__)
    notebooks = arguments['NOTEBOOKS'] or sorted(
        glob.glob('notebooks/**/*.ipynb', recursive=True)
    )

    starter_cell = get_starter_cell()
    for nb_path in notebooks:
        nb = read_notebook(nb_path)

        if should_update_nb(nb):
            nb = update_first_cell(nb, starter_cell)
            write_notebook(nb, nb_path)
            print('Wrote {}'.format(nb_path))
        else:
            print(
                'Skipping {} since its first cell is not code'.format(nb_path)
            )
