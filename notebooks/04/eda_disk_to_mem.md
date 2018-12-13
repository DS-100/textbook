

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

## From Disk to Memory

### Filesizes

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

Many computers have much more disk storage than available memory. It is not uncommon to have a data file happily stored on a computer that will overflow the computer's memory if we attempt to manipulate it with a program, including Python programs. In order to begin a data analysis, we often begin by making sure the files we will work with are of manageable size. To accomplish this, we use **command-line interface (CLI) tools**. The tools we introduce in this section have the following starting and ending constraints:

**Starting constraint:** Data files reside in disk storage.

**Ending constraint:** Data files are small enough to read into memory by a Python program.

That is, these tools assume that the files of interest are stored on disk. We can then use these tools to check whether we can likely read our data files into memory.

To use CLI tools, we enter commands into a shell interpreter such as `sh` or `bash`. These interpreters behave like the Python interpreter but have their own language, syntax, and built-in commands. We only cover a few useful commands in this section.

**Note:** all CLI tools we cover in this book are specific to the `sh` shell interpreter, the default interpreter for Jupyter installations on MacOS and Linux systems at the time of writing. Notebooks launched on Data 100's JupyterHub will also use the `sh` shell interpreter. Windows systems have a different interpreter and the commands shown in the book may not run on Windows, although Windows gives access to a `sh` interpreter through its Linux Subsystem.

Commonly, we open a terminal program to start a shell interpreter. Jupyter notebooks, however, provide a convenience: if a line of code is prefixed with the `!` character, the line will go directly to a shell interpreter. For example, the `ls` command lists the files in the current directory.


```python
!ls
```

    [34mdata[m[m                   eda_granularity.ipynb  eda_structure.ipynb
    eda_disk_to_mem.ipynb  eda_intro.ipynb        eda_temp.ipynb
    eda_faithfulness.ipynb eda_scope.ipynb


In the line above, Jupyter runs the `ls` command through a shell interpreter and displays the results of the command in the notebook.

CLI tools like `ls` often take in an **argument**, similar to how Python functions take in arguments. In `sh`, however, we wrap arguments with spaces, not parentheses. Calling `ls` with a folder as an argument shows all the files in the folder.


```python
!ls data
```

    [34mSFBusinesses[m[m  babies.data   babies23.data text.txt
    SFHousing.csv babies.readme stops.json



```python
!ls data/SFBusinesses/
```

    businesses.csv  inspections.csv legend.csv      violations.csv


Command-line tools like `ls` often support **flags** that provide additional options for the user. For example, adding the `-l` flag lists one file per line with additional metadata.


```python
!ls -l data
```

    total 114960
    drwx------@ 6 sam  staff       192 Dec 10 13:38 [34mSFBusinesses[m[m
    -rw-r--r--@ 1 sam  staff  51696074 Dec  9 19:52 SFHousing.csv
    -rw-r--r--@ 1 sam  staff     34654 Dec  9 19:52 babies.data
    -rw-r--r--@ 1 sam  staff      2040 Dec 10 11:08 babies.readme
    -rw-r--r--@ 1 sam  staff    142263 Dec 10 11:08 babies23.data
    -rw-r--r--  1 sam  staff   6124111 Aug 10 22:35 stops.json
    -rw-r--r--  1 sam  staff       177 Dec 12 20:24 text.txt


In particular, the fifth column of the listing shows the file size in bytes. For example, we can see that the `SFHousing.csv` file takes up `51696074` bytes on disk. To make the file sizes more readable, we can use the `-h` flag.


```python
!ls -l -h data
```

    total 114960
    drwx------@ 6 sam  staff   192B Dec 10 13:38 [34mSFBusinesses[m[m
    -rw-r--r--@ 1 sam  staff    49M Dec  9 19:52 SFHousing.csv
    -rw-r--r--@ 1 sam  staff    34K Dec  9 19:52 babies.data
    -rw-r--r--@ 1 sam  staff   2.0K Dec 10 11:08 babies.readme
    -rw-r--r--@ 1 sam  staff   139K Dec 10 11:08 babies23.data
    -rw-r--r--  1 sam  staff   5.8M Aug 10 22:35 stops.json
    -rw-r--r--  1 sam  staff   177B Dec 12 20:24 text.txt


We see that the `SFHousing.csv` file takes up 49 MiB on disk, making it well within the memory capacities of most systems.

**Folder Sizes**

Sometimes we are interested in the total size of a folder instead of individual files. For example, if we have one file of sensor recordings for each month in a year, we might like to see whether we combine all the data into a single DataFrame. Note that `ls` does not calculate folder sizes correctly. Notice `ls` shows that the `SFBusinesses` folder takes up 192 bytes on disk.


```python
!ls -l -h data
```

    total 114960
    drwx------@ 6 sam  staff   192B Dec 10 13:38 [34mSFBusinesses[m[m
    -rw-r--r--@ 1 sam  staff    49M Dec  9 19:52 SFHousing.csv
    -rw-r--r--@ 1 sam  staff    34K Dec  9 19:52 babies.data
    -rw-r--r--@ 1 sam  staff   2.0K Dec 10 11:08 babies.readme
    -rw-r--r--@ 1 sam  staff   139K Dec 10 11:08 babies23.data
    -rw-r--r--  1 sam  staff   5.8M Aug 10 22:35 stops.json
    -rw-r--r--  1 sam  staff   177B Dec 12 20:24 text.txt


However, the folder itself contains files that are larger than 192 bytes:


```python
!ls -l -h data/SFBusinesses/
```

    total 9496
    -rw-r--r--@ 1 sam  staff   645K Jan 26  2018 businesses.csv
    -rw-r--r--@ 1 sam  staff   455K Jan 26  2018 inspections.csv
    -rw-r--r--@ 1 sam  staff   120B Jan 26  2018 legend.csv
    -rw-r--r--@ 1 sam  staff   3.6M Jan 26  2018 violations.csv


To calculate the total size of a folder, including files in the folder, we use the `du` (short for disk usage) CLI tool. By default, the `du` tool shows the sizes of folders in bytes.


```python
!du data
```

    9496	data/SFBusinesses
    124472	data


To show file sizes using prefix notation we add the `-h` flag.


```python
!du -h data/SFBusinesses
```

    4.6M	data/SFBusinesses


We commonly also add the `-s` flag to `du` to show the file sizes for both files and folders. The asterisk in `data/*` below tells `du` to show the size of every item in the `data/*` folder.


```python
!du -sh data/*
```

    4.6M	data/SFBusinesses
     50M	data/SFHousing.csv
     36K	data/babies.data
    4.0K	data/babies.readme
    140K	data/babies23.data
    5.8M	data/stops.json
    4.0K	data/text.txt


**Memory Overhead**

As a rule of thumb, reading in a file using `pandas` usually requires at least double the available memory as the file size. That is, reading in a 1 GiB file will typically require at least 2 GiB of available memory.

Note that memory is shared by all programs running on a computer, including the operating system, web browsers, and yes, Jupyter notebook itself. A computer with 4 GiB total RAM might have only 1 GiB available RAM with many applications running. With 1 GiB available RAM, it is unlikely that `pandas` will be able to read in a 1 GiB file.

### Structure

A data file's **structure** refers to the format that the data are stored in. Thus far, we have worked exclusively with **tabular data**, data arranged in rows and columns. As you may imagine, not all data have such a convenient format. Common data file structures include:

- Comma-Separated Values (CSV) and Tab-Separated Values (TSV). These files contain tabular data with fields delimited by either a comma for CSV or a tab character (`\t`) for TSV.
- JavaScript Object Format (JSON), a common data format used for communication by web servers, has a hierarchical structure with keys and values similar to a Python dictionary.
- eXtensible Markup Language (XML) or HyperText Markup Language (HTML). These files also contain data in a nested format. We cover this format in greater detail later in the book.

Of course, there are a wealth of tools for working with data in various formats. In this book, however, we will almost always work with tabular data, converting from other formats as necessary. Why restrict ourselves in this way? First, much research has studied how to best store and manipulate tabular data. This has resulted in stable and efficient tools for working with tabular data. Second, tabular data are close cousins of matrices, the mathematical objects of the immensely rich field of linear algebra. Finally, tabular data are very common.

After verifying that a data file is small enough to hold in memory, we next determine whether the file has a tabular format. If the file has a tabular format, we can use the `pd.read_csv` method to quickly read the file into memory. Otherwise, we will have to turn to a tool that can work with the data format. For this part of EDA, we have the following starting and ending constraints:

**Starting constraint:** The data file resides on disk storage.

**Ending constraint:** The data file has a well-defined structure, ideally a tabular structure.

Assuming we have data files on disk, we can once again turn to command-line tools to check structure. The `head` command displays the first few lines of a file and is very useful for peeking at a file's contents.


```python
!head data/babies.data
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
!tail data/babies.data
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
    


We can print the entire file's contents using the `cat` command. Take care when using this command, however, as printing a large file can overflow the computer's memory.


```python
# This file is small, so using cat is safe.
!cat data/text.txt
```

    "city","zip","street"
    "Alameda","94501","1220 Broadway"
    "Alameda","94501","429 Fair Haven Road"
    "Alameda","94501","2804 Fernside Boulevard"
    "Alameda","94501","1316 Grove Street"

In many cases, using `head` and `tail` alone gives us a sense of the file structure. For example, we can see that the `babies.data` file contains tabular data using spaces as a delimiter.


```python
!head data/babies.data
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


Since the file is small, we can easily read in this data using `pandas`.


```python
# delimiter='\s+' says that each field is delimited by one or more spaces.
pd.read_csv('data/babies.data', delimiter='\s+')
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
      <th>bwt</th>
      <th>gestation</th>
      <th>parity</th>
      <th>age</th>
      <th>height</th>
      <th>weight</th>
      <th>smoke</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>120</td>
      <td>284</td>
      <td>0</td>
      <td>27</td>
      <td>62</td>
      <td>100</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>113</td>
      <td>282</td>
      <td>0</td>
      <td>33</td>
      <td>64</td>
      <td>135</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>128</td>
      <td>279</td>
      <td>0</td>
      <td>28</td>
      <td>64</td>
      <td>115</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1233</th>
      <td>130</td>
      <td>291</td>
      <td>0</td>
      <td>30</td>
      <td>65</td>
      <td>150</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1234</th>
      <td>125</td>
      <td>281</td>
      <td>1</td>
      <td>21</td>
      <td>65</td>
      <td>110</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1235</th>
      <td>117</td>
      <td>297</td>
      <td>0</td>
      <td>38</td>
      <td>65</td>
      <td>129</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>1236 rows × 7 columns</p>
</div>



## Summary

We have introduced the command-line tools `ls`, `du`, `head`, and `tail`. These tools help us verify the following constraints on data files:

**Starting constraint:** Data files reside on disk.

**Ending constraints:** Data files are small enough to read into memory and have a well-defined structure.

Completing this step of exploratory data analysis gives us confidence that we can use Python and the `pandas` package to proceed with the next steps of analysis.
