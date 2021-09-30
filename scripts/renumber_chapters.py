'''
Renames chapter folders so that the numbering is always incrementing starting
from 01.

Jupyter Book automatically numbers chapters in the order they appear in
_toc.yml, but we also number the folder names, e.g. ch/05/sql_subsetting. When
we edit _toc.yml, though, the folder names might get out of date with the
Jupyter Book numbering. This script renames folders in ch/ and updates _toc.yml
so that the folder numbering always matches what Jupyter Book outputs.

Warning: this breaks old URLs! When ch/05/sql_subsetting becomes
ch/07/sql_subsetting, existing URLs will break. Use this script with caution...

By default, the script outputs a list of changes without actually changing the
contents of ch/ or _toc.yml. Use the -c option to commit the changes.

Usage:
  renumber_chapters.py [-c]

Options:
  -c --commit  Renames folders in ch/ and paths in _toc.yml
'''
# %%
from pathlib import Path
from docopt import docopt
from ruamel.yaml import YAML
from collections import namedtuple

# Use RoundTrip loader to preserve old formatting
yaml = YAML(typ='rt')

content_path = Path('.') / 'content'
ch_path = content_path / 'ch'
toc_path = content_path / '_toc.yml'


# %%
def load_toc():
    with toc_path.open() as f:
        return yaml.load(f)


toc = load_toc()

TocLoc = namedtuple('TocLoc', 'part, ch')


def chapter(tocloc):
    part, ch = tocloc
    return toc['parts'][part]['chapters'][ch]


def get_folder(chap):
    intro_file = chap['file']
    [_, folder_name, *_] = intro_file.split('/')
    return folder_name


def set_folder(chap, new_folder):
    def set_path(path):
        parts = path.split('/')
        parts[1] = new_folder
        return '/'.join(parts)

    chap['file'] = set_path(chap['file'])
    for section in chap.get('sections', []):
        section['file'] = set_path(section['file'])
    return chap


# %%
def make_folder_counter():
    num = 1
    while True:
        yield f'{num:02}'
        num += 1


Change = namedtuple('Change', 'loc, current, desired')


def all_changes():
    changes = []
    new_folders = make_folder_counter()
    for part_pos, part in enumerate(toc['parts']):
        if not part.get('numbered'):
            continue
        for ch_pos, chap in enumerate(part['chapters']):
            loc = TocLoc(part_pos, ch_pos)
            current = get_folder(chap)
            desired = next(new_folders)
            if current != desired:
                changes.append(Change(loc, current, desired))
    return changes


changes = all_changes()
changes


# %%
def process_toc_changes(changes):
    '''Danger! Irreversibly changes _toc.yml'''
    for loc, _, desired in changes:
        set_folder(chapter(loc), desired)
    with toc_path.open('w') as f:
        yaml.dump(toc, f)


# %%
def process_folder_changes(changes):
    '''Danger! Irreversibly changes ch/ folders'''
    # Process in two passes since we need to avoid folder name conflicts
    temp_suffix = '__temp'

    for _, current, desired in changes:
        folder = ch_path / current
        folder.rename(ch_path / (desired + temp_suffix))

    for _, _, desired in changes:
        temp_folder = ch_path / (desired + temp_suffix)
        temp_folder.rename(ch_path / desired)


# %%
def renumber_chapters(commit=False):
    changes = all_changes()
    print(f'Will renumber these {len(changes)} folders:')
    for _, current, desired in changes:
        print(f'  {current:>4} -> {desired:<4}')

    if not commit:
        print('Rerun with -c to commit changes to disk')
        return

    print('Renaming folders in content/ch/...')
    process_folder_changes(changes)

    print('Updating content/_toc.yml...')
    process_toc_changes(changes)

    print('Done!')


# %%
if __name__ == '__main__':
    arguments = docopt(__doc__, version='1.0')
    renumber_chapters(arguments['--commit'])
