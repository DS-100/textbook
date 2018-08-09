"""Usage: convert_notebooks_to_html_partial.py [NOTEBOOKS ...]

This script takes the .ipynb files in the notebooks/ folder and removes the
hidden cells as well as the newlines before closing </div> tags so that the
resulting HTML partial can be embedded in a Gitbook page easily.

For reference:
https://nbconvert.readthedocs.org/en/latest/nbconvert_library.html
http://nbconvert.readthedocs.org/en/latest/nbconvert_library.html#using-different-preprocessors

Arguments:
    NOTEBOOKS  Notebook files to convert. By default, will convert all
               notebooks in the notebooks/ folder.
"""

import glob
import re
import os
from textwrap import dedent
from docopt import docopt
from subprocess import run

import nbformat
from nbinteract import InteractExporter
from traitlets.config import Config

wrapper = """
<div id="ipython-notebook">
    <div class="buttons">
        <button class="interact-button js-nbinteract-widget">
            Show Widgets
        </button>
        <a class="interact-button" href="{interact_link}">Open on DataHub</a>
    </div>
    {html}
</div>
"""

html_exporter = InteractExporter(
    config=Config(
        InteractExporter=dict(
            # Use ExtractOutputPreprocessor to extract images to separate files
            preprocessors=[
                'nbconvert.preprocessors.ExtractOutputPreprocessor'
            ],
            template_file='gitbook',
            button_at_top=False,
        )
    )
)

# Output notebook HTML partials into this directory
NOTEBOOK_HTML_DIR = 'notebooks-html'

# Output notebook HTML images into this directory
NOTEBOOK_IMAGE_DIR = 'notebooks-images'

# The prefix for the interact button links. The path format string gets filled
# in with the notebook as well as any datasets the notebook requires.
INTERACT_LINK = (
    'http://data100.datahub.berkeley.edu/user-redirect/git-pull'
    '?repo=https://github.com/DS-100/textbook&{paths}'
)

# The prefix for each notebook + its dependencies
PATH_PREFIX = 'subPath={}'

# Used to ensure all the closing div tags are on the same line for Markdown to
# parse them properly
CLOSING_DIV_REGEX = re.compile('\s+</div>')


def convert_notebooks_to_html_partial(notebook_paths):
    """
    Converts notebooks in notebook_paths to HTML partials in NOTEBOOK_HTML_DIR
    """
    for notebook_path in notebook_paths:
        # Computes <name>.ipynb from notebooks/01/<name>.ipynb
        path, filename = os.path.split(notebook_path)
        # Computes 01 from notebooks/01
        _, chapter = os.path.split(path)
        # Computes <name> from <name>.ipynb
        basename, _ = os.path.splitext(filename)
        # Computes <name>.html from notebooks/<name>.ipynb
        outfile_name = basename + '.html'

        # This results in images like AB_5_1.png for a notebook called AB.ipynb
        unique_image_key = basename
        # This sets the img tag URL in the rendered HTML. This restricts the
        # the chapter markdown files to be one level deep. It isn't ideal, but
        # the only way around it is to buy a domain for the staging textbook as
        # well and we'd rather not have to do that.
        output_files_dir = '/' + NOTEBOOK_IMAGE_DIR

        extract_output_config = {
            'unique_key': unique_image_key,
            'output_files_dir': output_files_dir,
        }

        notebook = nbformat.read(notebook_path, 4)
        notebook.cells.insert(0, _preamble_cell(path))
        html, resources = html_exporter.from_notebook_node(
            notebook,
            resources=extract_output_config,
        )

        with_wrapper = wrapper.format(
            interact_link=INTERACT_LINK.format(
                paths=PATH_PREFIX.format(notebook_path)
            ),
            html=html,
        )

        # Remove newlines before closing div tags
        final_output = CLOSING_DIV_REGEX.sub('</div>', with_wrapper)

        # Write out HTML
        outfile_path = os.path.join('ch', chapter, outfile_name)
        with open(outfile_path, 'w', encoding='utf-8') as outfile:
            outfile.write(final_output)

        # Write out images
        for relative_path, image_data in resources['outputs'].items():
            image_name = os.path.basename(relative_path)
            final_image_path = os.path.join(NOTEBOOK_IMAGE_DIR, image_name)
            with open(final_image_path, 'wb') as outimage:
                outimage.write(image_data)

        print(outfile_path + " written.")


def _preamble_cell(path):
    """
    This cell is inserted at the start of each notebook to set the working
    directory to the correct folder.
    """
    code = dedent(
        '''
    # HIDDEN
    # Clear previously defined variables
    %reset -f

    # Set directory for data loading to work properly
    import os
    os.chdir(os.path.expanduser('~/{}'))
    '''.format(path)
    )
    return nbformat.v4.new_code_cell(source=code)


def convert_notebooks_to_markdown(notebook_paths):
    """
    Since notebooks are difficult to use for code reviewing, this converts each
    notebook to Markdown so that reviewers can use the plain text versions of
    notebooks.
    """
    run(['jupyter', 'nbconvert', '--to', 'markdown', *notebook_paths])


if __name__ == '__main__':
    arguments = docopt(__doc__)
    notebooks = arguments['NOTEBOOKS'] or glob.glob(
        'notebooks/**/*.ipynb', recursive=True
    )
    os.makedirs(NOTEBOOK_HTML_DIR, exist_ok=True)
    os.makedirs(NOTEBOOK_IMAGE_DIR, exist_ok=True)
    convert_notebooks_to_html_partial(notebooks)
    convert_notebooks_to_markdown(notebooks)
