

```python
# HIDDEN
import warnings
# Ignore numpy dtype warnings. These warnings are caused by an interaction
# between numpy and Cython and can be safely ignored.
# Reference: https://stackoverflow.com/a/40846742
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
%matplotlib inline
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, interact_manual
import nbinteract as nbi

sns.set()
sns.set_context('talk')
np.set_printoptions(threshold=20, precision=2, suppress=True)
pd.options.display.max_rows = 7
pd.options.display.max_columns = 8
pd.set_option('precision', 2)
# This option stops scientific notation for pandas
# pd.set_option('display.float_format', '{:.2f}'.format)
```

## Structure

A dataset's **structure** refers to the arrangement of data in the dataset. `pandas` is most suited for working with **tabular data**, data arranged in rows and columns. As you may imagine, not all data have such a convenient structure.

Data files are stored in specific formats. For example, we have seen the Comma-Separated Values (`.csv`) file format, where each record is stored as a line in the file and each field is separated by a comma. Some file formats are more commonly used to store tabular data than others. In this book, we eventually introduce the following file formats:

- Comma-Separated Values (CSV) and Tab-Separated Values (TSV). These files contain tabular data with fields delimited by either a comma for CSV or a tab character (`\t`) for TSV.
- JavaScript Object Format (JSON) is a common data format used for communication by web servers. JSON files have a hierarchical structure with keys and values similar to a Python dictionary.
- eXtensible Markup Language (XML) or HyperText Markup Language (HTML). These files also contain data in a hierarchical format.

Of course, there are a wealth of tools for working with data in various formats. In this book, however, we will almost always work with tabular data, converting from other formats as necessary. Why restrict ourselves in this way? First, much research has studied how to best store and manipulate tabular data. This has resulted in stable and efficient tools for working with tabular data. Second, tabular data are close cousins of matrices, the mathematical objects of the immensely rich field of linear algebra. Finally, tabular data are very common.

If the file has a tabular format, we can use the `pd.read_csv` method to quickly read the file into memory. Otherwise, we will have to use another tool that can work with the data format.

To check a data file's format, we use **command-line interface (CLI) tools**. To use CLI tools, we enter commands into a shell interpreter such as `sh` or `bash`. These interpreters behave like the Python interpreter but have their own language, syntax, and built-in commands. Since many of the commands are less relevant for data science, we only cover a few useful commands in this section.

**Note:** all CLI tools we cover in this book are specific to the `sh` shell interpreter, the default interpreter for Jupyter installations on MacOS and Linux systems at the time of writing. Notebooks launched on Data 100's JupyterHub will also use the `sh` shell interpreter. Windows systems have a different interpreter and the commands shown in the book may not run on Windows, although Windows gives access to a `sh` interpreter through its Linux Subsystem.

Commonly, we open a terminal program to start a shell interpreter. Jupyter notebooks, however, provide a convenience: if a line of code is prefixed with the `!` character, the line will go directly to the system's shell interpreter. For example, the `ls` command lists the files in the current directory.


```python
!ls
```

    babynames.csv                       pandas_indexes.ipynb
    [34mothers[m[m                              pandas_intro.ipynb
    pandas_apply_strings_plotting.ipynb pandas_structure.ipynb
    pandas_grouping_pivoting.ipynb


In the line above, Jupyter runs the `ls` command through the `sh` shell interpreter and displays the results of the command in the notebook.

CLI tools like `ls` often take in an **argument**, similar to how Python functions take in arguments. In `sh`, however, we wrap arguments with spaces, not parentheses. Calling `ls` with a folder as an argument shows all the files in the folder.


```python
!ls others
```

    babies.data


Once we locate a file of interest, we use other command-line tools to check structure. The `head` command displays the first few lines of a file and is very useful for peeking at a file's contents.


```python
!head others/babies.data
```

    bwt gestation parity age height weight smoke
    120 284   0  27  62 100   0
    113 282   0  33  64 135   0
    128 279   0  28  64 115   1
    123 999   0  36  69 190   0
    108 282   0  23  67 125   1
    136 286   0  25  62  93   0
    138 244   0  33  62 178   0
    132 245   0  23  65 140   0
    120 289   0  25  62 125   0


By default, `head` displays the first 10 lines of a file. To display the last 10 lines, we use the `tail` command.


```python
!tail others/babies.data
```

    103 278   0  30  60  87   1
    118 276   0  34  64 116   0
    127 290   0  27  65 121   0
    132 270   0  27  65 126   0
    113 275   1  27  60 100   0
    128 265   0  24  67 120   0
    130 291   0  30  65 150   1
    125 281   1  21  65 110   0
    117 297   0  38  65 129   0
    


We can print the entire file's contents using the `cat` command. Take care when using this command, however, as printing a large file can cause the browser to crash.


```python
# This file is small, so using cat is safe.
!cat others/text.txt
```

    "city","zip","street"
    "Alameda","94501","1220 Broadway"
    "Alameda","94501","429 Fair Haven Road"
    "Alameda","94501","2804 Fernside Boulevard"
    "Alameda","94501","1316 Grove Street"

In many cases, using `head` and `tail` alone gives us a sense of the file structure. For example, we can see that the `babynames.csv` file contains tabular data in the CSV file format.


```python
!head babynames.csv
```

    Name,Sex,Count,Year
    Mary,F,9217,1884
    Anna,F,3860,1884
    Emma,F,2587,1884
    Elizabeth,F,2549,1884
    Minnie,F,2243,1884
    Margaret,F,2142,1884
    Ida,F,1882,1884
    Clara,F,1852,1884
    Bertha,F,1789,1884


We can easily read in CSV files using `pandas` using the `pd.read_csv` command.


```python
# pd is a common alias for pandas. We will always use the pd alias in this book
import pandas as pd

pd.read_csv('babynames.csv')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Sex</th>
      <th>Count</th>
      <th>Year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Mary</td>
      <td>F</td>
      <td>9217</td>
      <td>1884</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Anna</td>
      <td>F</td>
      <td>3860</td>
      <td>1884</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Emma</td>
      <td>F</td>
      <td>2587</td>
      <td>1884</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1891891</th>
      <td>Verna</td>
      <td>M</td>
      <td>5</td>
      <td>1883</td>
    </tr>
    <tr>
      <th>1891892</th>
      <td>Winnie</td>
      <td>M</td>
      <td>5</td>
      <td>1883</td>
    </tr>
    <tr>
      <th>1891893</th>
      <td>Winthrop</td>
      <td>M</td>
      <td>5</td>
      <td>1883</td>
    </tr>
  </tbody>
</table>
<p>1891894 rows × 4 columns</p>
</div>



### Filesizes

Notice that reading in the `babynames.csv` file results in a DataFrame with nearly two million rows. As of this writing, all computers have finite limits on computing power. You have likely encountered there limits firsthand if your computer has slowed down from having too many applications opened at once.

We often begin data analysis with datasets downloaded from the Internet. These files reside on the computer's **disk storage**. In order to use Python to explore and manipulate the data, however, we need to read the data into the computer's **memory**, also known as random access memory (RAM).

Unfortunately, a computer's RAM is typically much smaller than a computer's disk storage. For example, one computer model released in 2018 had 32 times more disk storage than RAM. This means that data files can often be much bigger than what is feasible to read into memory.

Both disk storage and RAM capacity are measured in terms of **bytes**. Roughly speaking, each character in a text file adds one byte to the file's size. For example, a file containing the following text has 177 characters and thus takes up 177 bytes of disk space.

    "city","zip","street"
    "Alameda","94501","1220 Broadway"
    "Alameda","94501","429 Fair Haven Road"
    "Alameda","94501","2804 Fernside Boulevard"
    "Alameda","94501","1316 Grove Street"

Of course, many of the datasets we work with today contain many characters. To succinctly describe the sizes of larger files, we use the following prefixes:

| Multiple | Notation | Number of Bytes    |
| -------- | -------- | ------------------ |
| Kibibyte | KiB      | 1024 = $ 2^{10} $  |
| Mebibyte | MiB      | 1024² = $ 2^{20} $ |
| Gibibyte | GiB      | 1024³ = $ 2^{30} $ |
| Tebibyte | TiB      | 1024⁴ = $ 2^{40} $ |
| Pebibyte | PiB      | 1024⁵ = $ 2^{50} $ |

For example, a file containing 52428800 characters takes up 52428800 bytes = 50 mebibytes = 50 MiB on disk.

Why use multiples of 1024 instead of simple multiples of 1000 for these prefixes? This is a historical result of the fact that nearly all computers use a binary number scheme where powers of 2 are simpler to represent. You will also see the typical SI prefixes used to describe size — kilobytes, megabytes, and gigabytes, for example. Unfortunately, these prefixes are used inconsistently. Sometimes a kilobyte refers to 1000 bytes; other times, a kilobyte refers to 1024 bytes. To avoid confusion, we will stick to kibi-, mebi-, and gibibytes which clearly represent multiples of 1024.

**When Is It Safe To Read In a File?**

Many computers have much more disk storage than available memory. It is not uncommon to have a data file happily stored on a computer that will overflow the computer's memory if we attempt to manipulate it with a program, including Python programs. In order to begin a data analysis, we often begin by making sure the files we will work with are of manageable size. To accomplish this, we use the command-line interface (CLI) tools `ls` and `du`.

Recall that `ls` shows the files within a folder:


```python
!ls others
```

    babies.data text.txt


Command-line tools like `ls` often support **flags** that provide additional options for the user. For example, adding the `-l` flag lists one file per line with additional metadata.


```python
!ls -l others
```

    total 80
    -rw-r--r--@ 1 sam  staff  34654 Dec 19 13:34 babies.data
    -rw-r--r--  1 sam  staff    177 Dec 19 13:37 text.txt


In particular, the fifth column of the listing shows the file size in bytes. For example, we can see that the `SFHousing.csv` file takes up `51696074` bytes on disk. To make the file sizes more readable, we can use the `-h` flag.


```python
!ls -l -h others
```

    total 80
    -rw-r--r--@ 1 sam  staff    34K Dec 19 13:34 babies.data
    -rw-r--r--  1 sam  staff   177B Dec 19 13:37 text.txt


We see that the `babies.data` file takes up 34 KiB on disk, making it well within the memory capacities of most systems. Although the `babynames.csv` file has nearly 2 million rows, it only takes up 30 MiB of disk storage. Most machines can read in the `babynames.csv` too.


```python
!ls -l -h
```

    total 62896
    -rw-r--r--  1 sam  staff    30M Aug 10 22:35 babynames.csv
    drwxr-xr-x  4 sam  staff   128B Dec 19 13:37 [34mothers[m[m
    -rw-r--r--  1 sam  staff   118K Sep 25 17:13 pandas_apply_strings_plotting.ipynb
    -rw-r--r--  1 sam  staff    34K Sep 25 17:13 pandas_grouping_pivoting.ipynb
    -rw-r--r--  1 sam  staff    32K Dec 19 13:07 pandas_indexes.ipynb
    -rw-r--r--  1 sam  staff   2.1K Dec 19 13:23 pandas_intro.ipynb
    -rw-r--r--  1 sam  staff    23K Dec 19 13:44 pandas_structure.ipynb


**Folder Sizes**

Sometimes we are interested in the total size of a folder instead of individual files. For example, if we have one file of sensor recordings for each month in a year, we might like to see whether we combine all the data into a single DataFrame. Note that `ls` does not calculate folder sizes correctly. Notice `ls` shows that the `others` folder takes up 128 bytes on disk.


```python
!ls -l -h 
```

    total 62896
    -rw-r--r--  1 sam  staff    30M Aug 10 22:35 babynames.csv
    drwxr-xr-x  4 sam  staff   128B Dec 19 13:37 [34mothers[m[m
    -rw-r--r--  1 sam  staff   118K Sep 25 17:13 pandas_apply_strings_plotting.ipynb
    -rw-r--r--  1 sam  staff    34K Sep 25 17:13 pandas_grouping_pivoting.ipynb
    -rw-r--r--  1 sam  staff    32K Dec 19 13:07 pandas_indexes.ipynb
    -rw-r--r--  1 sam  staff   2.1K Dec 19 13:23 pandas_intro.ipynb
    -rw-r--r--  1 sam  staff    23K Dec 19 13:44 pandas_structure.ipynb


However, the folder itself contains files that are larger than 128 bytes:


```python
!ls -l -h others
```

    total 80
    -rw-r--r--@ 1 sam  staff    34K Dec 19 13:34 babies.data
    -rw-r--r--  1 sam  staff   177B Dec 19 13:37 text.txt


To properly calculate the total size of a folder, including files in the folder, we use the `du` (short for disk usage) CLI tool. By default, the `du` tool shows the sizes of folders in its own units called blocks.


```python
!du others
```

    80	others


To show file sizes in bytes, we add the `-h` flag.


```python
!du -h others
```

     40K	others


We commonly also add the `-s` flag to `du` to show the file sizes for both files and folders. The asterisk in `others/*` below tells `du` to show the size of every item in the `others/*` folder.


```python
!du -sh others/*
```

     36K	others/babies.data
    4.0K	others/text.txt


**Memory Overhead**

As a rule of thumb, reading in a file using `pandas` usually requires at least double the available memory as the file size. That is, reading in a 1 GiB file will typically require at least 2 GiB of available memory.

Note that memory is shared by all programs running on a computer, including the operating system, web browsers, and yes, Jupyter notebook itself. A computer with 4 GiB total RAM might have only 1 GiB available RAM with many applications running. With 1 GiB available RAM, it is unlikely that `pandas` will be able to read in a 1 GiB file.

## Summary

We have introduced tabular data, the dataset structure that we use throughout the book. We have also introduced the command-line tools `ls`, `du`, `head`, and `tail`. These tools help us understand the format and structure of data files. We also use these tools to ensure that the data file is small enough to read into `pandas`. Once a file is read into `pandas`, we have a DataFrame that we use to proceed with analysis.
