
<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#Regular-Expressions" data-toc-modified-id="Regular-Expressions-1">Regular Expressions</a></span></li><li><span><a href="#Motivation" data-toc-modified-id="Motivation-2">Motivation</a></span></li><li><span><a href="#Regex-Syntax" data-toc-modified-id="Regex-Syntax-3">Regex Syntax</a></span><ul class="toc-item"><li><span><a href="#Literals" data-toc-modified-id="Literals-3.1">Literals</a></span></li><li><span><a href="#Wildcard-Character" data-toc-modified-id="Wildcard-Character-3.2">Wildcard Character</a></span></li><li><span><a href="#Character-Classes" data-toc-modified-id="Character-Classes-3.3">Character Classes</a></span></li><li><span><a href="#Negated-Character-Classes" data-toc-modified-id="Negated-Character-Classes-3.4">Negated Character Classes</a></span></li><li><span><a href="#Quantifiers" data-toc-modified-id="Quantifiers-3.5">Quantifiers</a></span></li><li><span><a href="#Anchoring" data-toc-modified-id="Anchoring-3.6">Anchoring</a></span></li><li><span><a href="#Escaping-Meta-Characters" data-toc-modified-id="Escaping-Meta-Characters-3.7">Escaping Meta Characters</a></span></li></ul></li><li><span><a href="#Reference-Tables" data-toc-modified-id="Reference-Tables-4">Reference Tables</a></span></li><li><span><a href="#Summary" data-toc-modified-id="Summary-5">Summary</a></span></li></ul></div>


```python
# HIDDEN
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
%matplotlib inline
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, interact_manual
import nbinteract as nbi
import re

sns.set()
sns.set_context('talk')
np.set_printoptions(threshold=20, precision=2, suppress=True)
pd.options.display.max_rows = 7
pd.options.display.max_columns = 8
pd.set_option('precision', 2)
# This option stops scientific notation for pandas
# pd.set_option('display.float_format', '{:.2f}'.format)
```

## Regular Expressions

In this section we introduce regular expressions, an important tool to specify patterns in strings.

## Motivation

In a larger piece of text, many useful substrings come in a specific format. For instance, the sentence below contains a U.S. phone number.

`"give me a call, my number is 123-456-7890."`

The phone number contains the following pattern:

1. Three numbers
1. Followed by a dash
1. Followed by three numbers
1. Followed by a dash
1. Followed by four Numbers

Given a free-form segment of text, we might naturally wish to detect and extract the phone numbers. We may also wish to extract specific pieces of the phone numbers—for example, by extracting the area code we may deduce the locations of individuals mentioned in the text.

To detect whether a string contains a phone number, we may attempt to write a method like the following:


```python
def is_phone_number(string):
    
    digits = '0123456789'
    
    def is_not_digit(token):
        return token not in digits 
    
    # Three numbers
    for i in range(3):
        if is_not_digit(string[i]):
            return False
    
    # Followed by a dash
    if string[3] != '-':
        return False
    
    # Followed by three numbers
    for i in range(4, 7):
        if is_not_digit(string[i]):
            return False
        
    # Followed by a dash    
    if string[7] != '-':
        return False
    
    # Followed by four numbers
    for i in range(8, 12):
        if is_not_digit(string[i]):
            return False
    
    return True
```


```python
is_phone_number("382-384-3840")
```




    True




```python
is_phone_number("phone number")
```




    False



The code above is unpleasant and verbose. Rather than manually loop through the characters of the string, we would prefer to specify a pattern and command Python to match the pattern.

**Regular expressions** (often abbreviated **regex**) conveniently solve this exact problem by allowing us to create general patterns for strings. Using a regular expression, we may re-implement the `is_phone_number` method in two short lines of Python:


```python
import re

def is_phone_number(string):
    regex = r"[0-9]{3}-[0-9]{3}-[0-9]{4}"
    return re.search(regex, string) is not None

is_phone_number("382-384-3840")
```




    True



In the code above, we use the regex `[0-9]{3}-[0-9]{3}-[0-9]{4}` to match phone numbers. Although cryptic at a first glance, the syntax of regular expressions is fortunately much simpler to learn than the Python language itself; we introduce nearly all of the syntax in this section alone.

We will also introduce the built-in Python module `re` that performs string operations using regexes. 

## Regex Syntax

We start with the syntax of regular expressions. In Python, regular expressions are most commonly stored as raw strings. Raw strings behave like normal Python strings without special handling for backslashes.

For example, to store the string `hello \ world` in a normal Python string, we must write:


```python
# Backslashes need to be escaped in normal Python strings
some_string = 'hello \\ world'
print(some_string)
```

    hello \ world


Using a raw string removes the need to escape the backslash:


```python
# Note the `r` prefix on the string
some_raw_string = r'hello \ world'
print(some_raw_string)
```

    hello \ world


Since backslashes appear often in regular expressions, we will use raw strings for all regexes in this section.

### Literals

A **literal** character in a regular expression matches the character itself. For example, the regex `r"a"` will match any `"a"` in `"Say! I like green eggs and ham!"`. All alphanumeric characters and most punctuation characters are regex literals.


```python
# HIDDEN
def show_regex_match(text, regex):
    """
    Prints the string with the regex match highlighted.
    """
    print(re.sub(f'({regex})', r'\033[1;30;43m\1\033[m', text))
```


```python
# The show_regex_match method highlights all regex matches in the input string
regex = r"green"
show_regex_match("Say! I like green eggs and ham!", regex)
```

    Say! I like [1;30;43mgreen[m eggs and ham!



```python
show_regex_match("Say! I like green eggs and ham!", r"a")
```

    S[1;30;43ma[my! I like green eggs [1;30;43ma[mnd h[1;30;43ma[mm!


In the example above we observe that regular expressions can match patterns that appear anywhere in the input string. In Python, this behavior differs depending on the method used to match the regex—some methods only return a match if the regex appears at the start of the string; some methods return a match anywhere in the string.

Notice also that the `show_regex_match` method highlights all occurrences of the regex in the input string. Again, this differs depending on the Python method used—some methods return all matches while some only return the first match.

Regular expressions are case-sensitive. In the example below, the regex only matches the lowercase `s` in `eggs`, not the uppercase `S` in `Say`.


```python
show_regex_match("Say! I like green eggs and ham!", r"s")
```

    Say! I like green egg[1;30;43ms[m and ham!


### Wildcard Character

Some characters have special meaning in a regular expression. These meta characters allow regexes to match a variety of patterns.

In a regular expression, the period character `.` matches any character except a newline.


```python
show_regex_match("Call me at 382-384-3840.", r".all")
```

    [1;30;43mCall[m me at 382-384-3840.


To match only the literal period character we must escape it with a backslash:


```python
show_regex_match("Call me at 382-384-3840.", r"\.")
```

    Call me at 382-384-3840[1;30;43m.[m


By using the period character to mark the parts of a pattern that vary, we construct a regex to match phone numbers. For example, we may take our original phone number `382-384-3840` and replace the numbers with `.`, leaving the dashes as literals. This results in the regex `...-...-....`.


```python
show_regex_match("Call me at 382-384-3840.", "...-...-....")
```

    Call me at [1;30;43m382-384-3840[m.


Since the period character matches all characters, however, the following input string will produce a spurious match.


```python
show_regex_match("My truck is not-all-blue.", "...-...-....")
```

    My truck is [1;30;43mnot-all-blue[m.


### Character Classes

A **character class** matches a specified set of characters, allowing us to create more restrictive matches than the `.` character alone. To create a character class, wrap the set of desired characters in brackets `[ ]`.


```python
show_regex_match("I like your gray shirt.", "gr[ae]y")
```

    I like your [1;30;43mgray[m shirt.



```python
show_regex_match("I like your grey shirt.", "gr[ae]y")
```

    I like your [1;30;43mgrey[m shirt.



```python
# Does not match; a character class only matches one character from a set
show_regex_match("I like your graey shirt.", "gr[ae]y")
```

    I like your graey shirt.



```python
# In this example, repeating the character class will match
show_regex_match("I like your graey shirt.", "gr[ae][ae]y")
```

    I like your [1;30;43mgraey[m shirt.


In a character class, the `.` character is treated as a literal, not as a wildcard.


```python
show_regex_match("I like your grey shirt.", "irt[.]")
```

    I like your grey sh[1;30;43mirt.[m


There are a few special shorthand notations we can use for commonly used character classes:

Shorthand | Meaning
--- | ---
[0-9] | All the digits
[a-z] | Lowercase letters
[A-Z] | Uppercase letters


```python
show_regex_match("I like your gray shirt.", "y[a-z]y")
```

    I like your gray shirt.


Character classes allow us to create a more specific regex for phone numbers.


```python
# We replaced every `.` character in ...-...-.... with [0-9] to restrict
# matches to digits.
phone_regex = r'[0-9][0-9][0-9]-[0-9][0-9][0-9]-[0-9][0-9][0-9][0-9]'
show_regex_match("Call me at 382-384-3840.", phone_regex)
```

    Call me at [1;30;43m382-384-3840[m.



```python
# Now we no longer match this string:
show_regex_match("My truck is not-all-blue.", phone_regex)
```

    My truck is not-all-blue.


### Negated Character Classes

A **negated character class** matches any character **except** the characters in the class. To create a negated character class, wrap the negated characters in `[^ ]`.


```python
show_regex_match("The car parked in the garage.", r"[^c]ar")
```

    The car [1;30;43mpar[mked in the [1;30;43mgar[mage.


### Quantifiers

To create a regex to match phone numbers, we wrote:

```
[0-9][0-9][0-9]-[0-9][0-9][0-9]-[0-9][0-9][0-9][0-9]
```

This matches 3 digits, a dash, 3 more digits, a dash, and 4 more digits.

Quantifiers allow us to match multiple consecutive appearances of a pattern. We specify the number of repetitions by placing the number in curly braces `{ }`.


```python
phone_regex = r'[0-9]{3}-[0-9]{3}-[0-9]{4}'
show_regex_match("Call me at 382-384-3840.", phone_regex)
```

    Call me at [1;30;43m382-384-3840[m.



```python
# No match
phone_regex = r'[0-9]{3}-[0-9]{3}-[0-9]{4}'
show_regex_match("Call me at 12-384-3840.", phone_regex)
```

    Call me at 12-384-3840.


A quantifier always modifies the character or character class to its immediate left. The following table shows the complete syntax for quantifiers.

Quantifier | Meaning
--- | ---
{m, n} | Match the preceding character m to n times.
{m} | Match the preceding character exactly m times.
{m,} | Match the preceding character at least m times.
{,n} | Match the preceding character at most n times.

**Shorthand Quantifiers**

Some commonly used quantifiers have a shorthand:

Symbol | Quantifier | Meaning
--- | --- | ---
* | {0,} | Match the preceding character 0 or more times
+ | {1,} | Match the preceding character 1 or more times
? | {0,1} | Match the preceding charcter 0 or 1 times

We use the `*` character instead of `{0,}` in the following examples.


```python
# 3 a's
show_regex_match('He screamed "Aaaah!" as the cart took a plunge.', "Aa*h!")
```

    He screamed "[1;30;43mAaaah![m" as the cart took a plunge.



```python
# Lots of a's
show_regex_match(
    'He screamed "Aaaaaaaaaaaaaaaaaaaah!" as the cart took a plunge.',
    "Aa*h!"
)
```

    He screamed "[1;30;43mAaaaaaaaaaaaaaaaaaaah![m" as the cart took a plunge.



```python
# No lowercase a's
show_regex_match('He screamed "Ah!" as the cart took a plunge.', "Aa*h!")
```

    He screamed "[1;30;43mAh![m" as the cart took a plunge.


**Quantifiers are greedy**

Quantifiers will always return the longest match possible. This sometimes results in surprising behavior:


```python
# We tried to match 311 and 911 but matched the ` and ` as well because
# `<311> and <911>` is the longest match possible for `<.+>`.
show_regex_match("Remember the numbers <311> and <911>", "<.+>")
```

    Remember the numbers [1;30;43m<311> and <911>[m


In many cases, using a more specific character class prevents these false matches:


```python
show_regex_match("Remember the numbers <311> and <911>", "<[0-9]+>")
```

    Remember the numbers [1;30;43m<311>[m and [1;30;43m<911>[m


### Anchoring

Sometimes a pattern should only match at the beginning or end of a string.  The special character `^` anchors the regex to match only if the pattern appears at the beginning of the string; the special character `$` anchors the regex to match only if the pattern occurs at the end of the string.  For example the regex `well$` only matches an appearance of `well` at the end of the string.


```python
show_regex_match('well, well, well', r"well$")
```

    well, well, [1;30;43mwell[m


Using both `^` and `$` requires the regex to match the full string.


```python
phone_regex = r"^[0-9]{3}-[0-9]{3}-[0-9]{4}$"
show_regex_match('382-384-3840', phone_regex)
```

    [1;30;43m382-384-3840[m



```python
# No match
show_regex_match('You can call me at 382-384-3840.', phone_regex)
```

    You can call me at 382-384-3840.


### Escaping Meta Characters

All regex meta characters have special meaning in a regular expression. To match meta characters as literals, we escape them using the `\` character.


```python
# `[` is a meta character and requires escaping
show_regex_match("Call me at [382-384-3840].", "\[")
```

    Call me at [1;30;43m[[m382-384-3840].



```python
# `.` is a meta character and requires escaping
show_regex_match("Call me at [382-384-3840].", "\.")
```

    Call me at [382-384-3840][1;30;43m.[m


## Reference Tables

We have now covered the most important pieces of regex syntax and meta characters. For a more complete reference, we include the tables below.

**Meta Characters**

This table includes most of the important *meta characters*, which help us specify certain patterns we want to match in a string.

| |Description|Example|Matches|Doesn't Match|
|---|---|---|---|---|
|.|Any character except \n|...|abc|ab<br>abcd|
|[ ]|Any character inside brackets|[cb.]ar|car<br>.ar|jar|
|[^ ]|Any character *not* inside brackets|[^b]ar|car<br>par|bar<br>ar|
|\*|≥ 0 or more of last symbol|[pb]\*ark|bbark<br>ark|dark|
|+|≥ 1 or more of last symbol|[pb]+ark|bbpark<br>bark|dark<br>ark|
|?|0 or 1 of last symbol|s?he|she<br>he|the|
|{*n*}|Exactly *n* of last symbol|hello{3}|hellooo|hello|
|&#124;|Pattern before or after bar|we&#124;[ui]s|we<br>us<br>is|e<br>s|
|\|Escapes next character|`\[hi\]`|[hi]|hi|
|^|Beginning of line|^ark|ark two|dark|
|\$|End of line|ark$|noahs ark|noahs arks|

**Shorthand Character Sets**

Some commonly used character sets have shorthands.

|Bracket Form|Shorthand|Description|
|---|---|---|
|[a-zA-Z0-9]|\\w|Alphanumeric character|
|[^a-zA-Z0-9]|\\W|Not an alphanumeric character|
|[0-9]|\d|Digit|
|[^0-9]|\D|Not a digit|
|[\t\n\f\r\p{Z}]|\s|Whitespace|
|[^\t\n\f\r\p{Z}]|\S|Not whitespace|

## Summary

Almost all programming languages have a library to match patterns using regular expressions, making them useful regardless of the specific language. In this section, we introduce regex syntax and the most useful meta characters.
