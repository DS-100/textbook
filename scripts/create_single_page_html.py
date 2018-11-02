"""
Using the generated HTML files created from
convert_notebooks_to_html_partial.py, creates a single long HTML page
containing all the textbook pages in the order defined in the table of
contents.

Once fed into Jekyll, this page will be styled and converted into a PDF using
the `ebook-convert` binary from Calibre.

This script also uses the `mjpage` binary to render math as SVGs
(https://github.com/pkra/mathjax-node-page/). It must be installed before
running this script.
"""
import os
from subprocess import check_call

from bs4 import BeautifulSoup
from markdown2 import markdown_path

from read_url_map import read_url_list

OUTPUT_FILENAME = 'book.html'
TEMP_FILENAME = 'book_temp.html'

RAW_START = '{% raw %}'
RAW_END = '{% endraw %}'

output_preamble = """---
layout: single_page
---

{% raw %}
"""

output_epilogue = """
{% endraw %}
"""

url_replacements = {
    '/': 'index.md',
    '/about_this_book.html': 'about_this_book.md'
}


def _replace_url(url):
    return (
        url.strip('/')
        if url not in url_replacements else url_replacements[url]
    )


def read_md_file(filename):
    return markdown_path(filename, extras=['metadata'])


def read_html_file(filename):
    with open(filename) as f:
        soup = BeautifulSoup(f, 'lxml')
        nb = soup.find(id='ipython-notebook')
        for button in nb.find_all(class_='buttons'):
            button.decompose()
        return str(nb)


def read_file(filename):
    if filename.endswith('.html'):
        return read_html_file(filename)
    elif filename.endswith('.md'):
        return read_md_file(filename)
    else:
        raise 'Unrecognized file type: ' + filename


def concat_pages(output_filename=OUTPUT_FILENAME):
    """
    Takes invidividual pages from the table of contents and concatenates them
    into a single, long HTML page.
    """
    files_to_read = list(map(_replace_url, read_url_list()))[:5]

    print('Concatenating book pages...')
    with open(output_filename, 'w') as f:
        f.write(output_preamble)

        for filename in files_to_read:
            f.write(read_file(filename))

        f.write(output_epilogue)
    print('Finished concatenating book pages.')
    print()


def render_math(output_filename=OUTPUT_FILENAME):
    """
    Uses the mjpage binary to render math in the textbook.
    """
    print('Rendering math (takes a couple minutes)...')
    with open(output_filename) as book, open(TEMP_FILENAME, 'w') as temp:
        check_call(['mjpage', '--dollars', '--format', '"TeX"'],
                   stdin=book,
                   stdout=temp)
    print('Finished rendering math.')
    print()

    print('Writing final HTML back to {}...'.format(output_filename))
    with open(TEMP_FILENAME) as temp, open(output_filename, 'w') as book:
        book.write(output_preamble)

        raw_seen = False
        for line in temp:
            if RAW_START in line:
                raw_seen = True
                continue
            elif RAW_END in line:
                break
            elif raw_seen:
                book.write(line)

        book.write(output_epilogue)
    os.remove(TEMP_FILENAME)
    print('Done!')


def fix_image_paths(output_filename=OUTPUT_FILENAME):
    """
    Replaces absolute image URLs with links to localhost so that ebook-convert
    can locate them.
    """
    check_call([
        'sed', '-i', "",
        "'s#/notebooks-images#http://localhost:4000/notebooks-images#g'",
        output_filename
    ])


if __name__ == '__main__':
    concat_pages()
    render_math()
    fix_image_paths()
