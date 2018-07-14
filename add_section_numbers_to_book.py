"""
This script takes the SUMMARY.md file and adds section numbers to each part.

This script is idempotent, so running it multiple times on the same file
produces the same output as running it once.

To re-number sections, you should change the sequence of the sections in the
SUMMARY.md file and rerun the script without manually changing the sections
themselves.

For example, given this `SUMMARY.md` file:

```
# Table of Contents

* [Read Me](/README.md)

<!-- begin_numbering -->
* [Introduction](/introduction/README.md)
  * [Motivation](/introduction/Motivation.md)
  * [Getting Started](/introduction/Getting_Started.md)
* [Tutorial](/tutorial/README.md)
  * [Simple Widgets](/tutorial/Simple_Widgets.md)
  * [Interact](/tutorial/Interact.md)
<!-- end_numbering -->

* [Appendix](/appendix.md)
```

This script will generate:

```
# Table of Contents

* [Read Me](/README.md)

<!-- begin_numbering -->
* [1. Introduction](/introduction/README.md)
  * [1.1 Motivation](/introduction/Motivation.md)
  * [1.2 Getting Started](/introduction/Getting_Started.md)
* [2. Tutorial](/tutorial/README.md)
  * [2.1 Simple Widgets](/tutorial/Simple_Widgets.md)
  * [2.2 Interact](/tutorial/Interact.md)
<!-- end_numbering -->

* [Appendix](/appendix.md)
```

To re-number sections, we can just swap sections in the SUMMARY.md:

```
# Table of Contents
* [1. Read Me](/README.md)
* [3. Tutorial](/tutorial/README.md)
  * [3.1 Simple Widgets](/tutorial/Simple_Widgets.md)
  * [3.2 Interact](/tutorial/Interact.md)
* [2. Introduction](/introduction/README.md)
  * [2.1 Motivation](/introduction/Motivation.md)
  * [2.2 Getting Started](/introduction/Getting_Started.md)
```

And the script will regenerate the correct section numbers:

```
# Table of Contents
* [1. Read Me](/README.md)
* [2. Tutorial](/tutorial/README.md)
  * [2.1 Simple Widgets](/tutorial/Simple_Widgets.md)
  * [2.2 Interact](/tutorial/Interact.md)
* [3. Introduction](/introduction/README.md)
  * [3.1 Motivation](/introduction/Motivation.md)
  * [3.2 Getting Started](/introduction/Getting_Started.md)
```
"""
import re
from codecs import open

# Name of the summary file
SUMMARY = 'SUMMARY.md'

# Indent amount in summary file
INDENT = '  '

# Regex to match any link
LINK_RE = re.compile(r'^\s*(\*|-|\+)')

# Regex to match previously labeled sections
LABEL_RE = re.compile(r'^[0-9\. ]+')

# Markers to begin and end section numbering
BEGIN = '<!-- begin_numbering -->'
END = '<!-- end_numbering -->'


def main():
    lines = read_summary()
    lines_with_labels = add_section_labels(lines)
    write_summary(lines_with_labels)
    print('Successfully labeled SUMMARY.md')


def add_section_labels(lines, section=(0, 0, 0), should_label=False):
    """
    Recursively adds the section label to each line. Removes previous labels
    before adding label to avoid duplication.
    """
    if len(lines) == 0:
        return lines

    line, *rest = lines
    part, subpart, subsubpart = section

    # Read section markers
    if is_begin(line):
        return [line, *add_section_labels(rest, section, should_label=True)]
    if is_end(line):
        return [line, *add_section_labels(rest, section, should_label=False)]

    if not should_label or not is_link(line):
        # Skip this line
        return [line, *add_section_labels(rest, section, should_label)]

    # Unfinished chapters aren't linked in the sidebar but we label them anyway
    if '[' not in line:
        before, after = line.split(' ', maxsplit=1)
        bracket = ' '
    else:
        before, after = line.split('[', maxsplit=1)
        bracket = '['

    after = LABEL_RE.sub('', after)

    # We only increment the numbering for a section *after* the recursion since
    # we can't tell ahead of time whether the next link is nested.
    if is_subsubpart(line):
        labeled_line = (
            f'{before}{bracket}{part}.{subpart}.{subsubpart + 1} {after}'
        )
        return [
            labeled_line, *add_section_labels(
                rest, (part, subpart, subsubpart + 1), should_label
            )
        ]
    elif is_subpart(line):
        labeled_line = f'{before}{bracket}{part}.{subpart + 1} {after}'
        return [
            labeled_line,
            *add_section_labels(rest, (part, subpart + 1, 0), should_label)
        ]
    else:
        labeled_line = f'{before}{bracket}{part + 1}. {after}'
        return [
            labeled_line,
            *add_section_labels(rest, (part + 1, 0, 0), should_label)
        ]


def read_summary():
    with open(SUMMARY, encoding='utf8') as f:
        return f.readlines()


def write_summary(lines: [str]):
    with open(SUMMARY, mode='w', encoding='utf8') as f:
        f.writelines(lines)


def is_link(line: str):
    return LINK_RE.match(line)


def is_subpart(line: str):
    return line.startswith(INDENT)


def is_subsubpart(line: str):
    return line.startswith(INDENT * 2)


def is_begin(line: str):
    return line == BEGIN


def is_end(line: str):
    return line == END


def test():
    test1 = [
        '# Table of Contents',
        '<!-- begin_numbering -->',
        '* [Read Me](/README.md)',
        '* [Introduction](/introduction/README.md)',
        '  * [Motivation](/introduction/Motivation.md)',
        '  * [Getting Started](/introduction/Getting_Started.md)',
        '* [Tutorial](/tutorial/README.md)',
        '  * [Simple Widgets](/tutorial/Simple_Widgets.md)',
        '  * [Interact](/tutorial/Interact.md)',
    ]

    res1 = [
        '# Table of Contents',
        '<!-- begin_numbering -->',
        '* [1. Read Me](/README.md)',
        '* [2. Introduction](/introduction/README.md)',
        '  * [2.1 Motivation](/introduction/Motivation.md)',
        '  * [2.2 Getting Started](/introduction/Getting_Started.md)',
        '* [3. Tutorial](/tutorial/README.md)',
        '  * [3.1 Simple Widgets](/tutorial/Simple_Widgets.md)',
        '  * [3.2 Interact](/tutorial/Interact.md)',
    ]

    test2 = [
        '# Table of Contents',
        '<!-- begin_numbering -->',
        '* [1. Read Me](/README.md)',
        '* [3. Tutorial](/tutorial/README.md)',
        '  * [3.1 Simple Widgets](/tutorial/Simple_Widgets.md)',
        '  * [3.2 Interact](/tutorial/Interact.md)',
        '* [2. Introduction](/introduction/README.md)',
        '  * [2.1 Motivation](/introduction/Motivation.md)',
        '  * [2.2 Getting Started](/introduction/Getting_Started.md)',
    ]

    res2 = [
        '# Table of Contents',
        '<!-- begin_numbering -->',
        '* [1. Read Me](/README.md)',
        '* [2. Tutorial](/tutorial/README.md)',
        '  * [2.1 Simple Widgets](/tutorial/Simple_Widgets.md)',
        '  * [2.2 Interact](/tutorial/Interact.md)',
        '* [3. Introduction](/introduction/README.md)',
        '  * [3.1 Motivation](/introduction/Motivation.md)',
        '  * [3.2 Getting Started](/introduction/Getting_Started.md)',
    ]

    test3 = [
        '# Table of Contents',
        '<!-- begin_numbering -->',
        '* [1. Read Me](/README.md)',
        '* [Read Me 2](/README.md)',
        '* Unfinished chapter',
    ]

    res3 = [
        '# Table of Contents',
        '<!-- begin_numbering -->',
        '* [1. Read Me](/README.md)',
        '* [2. Read Me 2](/README.md)',
        '* 3. Unfinished chapter',
    ]

    test4 = [
        '# Table of Contents',
        '* [Should not number me](foo.md)',
        '<!-- begin_numbering -->',
        '* [1. Read Me](/README.md)',
        '* [Read Me 2](/README.md)',
        '* Unfinished chapter',
        '<!-- end_numbering -->',
        '* [Appendix](appendix.md)',
    ]

    res4 = [
        '# Table of Contents',
        '* [Should not number me](foo.md)',
        '<!-- begin_numbering -->',
        '* [1. Read Me](/README.md)',
        '* [2. Read Me 2](/README.md)',
        '* 3. Unfinished chapter',
        '<!-- end_numbering -->',
        '* [Appendix](appendix.md)',
    ]

    assert add_section_labels(test1) == res1
    assert add_section_labels(test2) == res2
    assert add_section_labels(test3) == res3
    assert add_section_labels(test4) == res4
    print('Tests pass')


if __name__ == '__main__':
    # To run tests, uncomment this line and comment the main() line
    # test()
    main()
