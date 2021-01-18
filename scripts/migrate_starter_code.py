'''
Overwrites the first cell of ALL notebooks with the cell in `starter.ipynb`.

If first cell of notebook isn't a code cell, adds a code will with cell in
starter.ipynb .

To check first three cell types of all notebooks:

    ls content/ch/**/*.ipynb -1 | \
        xargs -I {} \
        jq -c "[input_filename, .cells[0].cell_type, .cells[1].cell_type , .cells[2].cell_type ]" {}

Usage: migrate_starter_code.py

TODO(sam): Write this script
'''
