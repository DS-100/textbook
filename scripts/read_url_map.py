"""
Exports a function `read_url_map(yaml_path)` that reads _data/toc.yml
and returns a mapping from each URL to its previous and next URLs in the
textbook. The ordered dictionary looks like:

    {
        'ch/01/some_page.html' : {
            'prev': 'about.html',
            'next': 'ch/01/foo.html',
        },
        ...
    }
"""
import yaml
import toolz.curried as t
from itertools import chain

__all__ = ['read_url_map', 'read_redirects']

# YAML file containing textbook table of contents
TOC_PATH = '_data/toc.yml'


def read_url_map(yaml_path=TOC_PATH) -> dict:
    """
    Generates mapping from each URL to its previous and next URLs in the
    textbook. The dictionary looks like:

    {
        'ch/01/some_page.html' : {
            'prev': 'about.html',
            'next': 'ch/01/foo.html',
        },
        ...
    }
    """
    return t.pipe(
        read_url_list(yaml_path),
        _sliding_three,
        t.map(_adj_pages),
        t.merge(),
    )


def read_redirects(yaml_path=TOC_PATH) -> dict:
    """
    Generates redirect mapping of old URL to new URL:

    {
        'ch/04/cleaning_intro.html': 'ch/05/cleaning_intro.html',
        ...
    }
    """
    with open(yaml_path) as f:
        data = yaml.load(f)

    return t.pipe(
        data,
        t.map(_get_redirects),
        t.filter(t.identity),
        t.concat,
        _merge_redirects,
    )


def read_url_list(yaml_path=TOC_PATH) -> list:
    """
    Generates flat list of section HTML names from the table of contents.
    List looks like:

    [
        '',
        'about_this_book.html',
        'ch/01/lifecycle_intro.html',
        ...
    ]
    """
    with open(yaml_path) as f:
        data = yaml.load(f)

    return t.pipe(
        data,
        t.remove(_not_internal_link),
        flatmap(_flatten_sections),
        t.map(t.get('url')),
        list,
    )


flatmap = t.curry(lambda f, items: chain.from_iterable(map(f, items)))


def _not_internal_link(entry):
    return not entry.get('url', '').startswith('/')


def _get_redirects(entry):
    return entry.get('redirects', False)


def _merge_redirects(redirects):
    def _merge_redirect(mapping, redirect):
        return t.assoc(mapping, redirect['from'], redirect['to'])

    return t.reduce(_merge_redirect, redirects, {})


def _flatten_sections(entry):
    sections = entry.get('sections', [])
    return [t.dissoc(entry, 'sections')] + sections


def _sliding_three(entries):
    return ([(None, entries[0], entries[1])] + list(
        t.sliding_window(3, entries)
    ) + [(entries[-2], entries[-1], None)])


wrap_url = "'{}'".format


def _adj_pages(triplet):
    prev, cur, nex = triplet
    return {
        cur.strip('/'): {
            'prev': wrap_url(prev) if prev is not None else 'false',
            'next': wrap_url(nex) if nex is not None else 'false',
        }
    }


if __name__ == '__main__':
    from pprint import pprint
    pprint(read_redirects())
