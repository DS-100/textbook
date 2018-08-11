
<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#The-Relational-Model" data-toc-modified-id="The-Relational-Model-1">The Relational Model</a></span></li><li><span><a href="#Relational-Database-Management-Systems" data-toc-modified-id="Relational-Database-Management-Systems-2">Relational Database Management Systems</a></span></li><li><span><a href="#RDBMS-vs.-pandas" data-toc-modified-id="RDBMS-vs.-pandas-3">RDBMS vs. pandas</a></span></li></ul></div>

## The Relational Model 

A **database** is an organized collection of data. In the past, data was stored in specialized data structures that were designed for specific tasks. For example, airlines might record flight bookings in a different format than a bank managing an account ledger. In 1969, Ted Codd introduced the relational model as a general method of storing data. Data is stored in two-dimensional tables called **relations**, consisting of individual observations in each row (commonly referred to as **tuples**). Each tuple is a structured data item that represents the relationship between certain **attributes** (columns). Each attribute of a relation has a name and data type.

Consider the `purchases` relation below:

<header style="text-align:center"><strong>purchases</strong></header>
<table border="1" class="DataFrame">
    <thead>
        <tr>
            <td><b>name</b></td>
            <td><b>product</b></td>
            <td><b>retailer</b></td>
            <td><b>date purchased</b></td>
        </tr>
    </thead>
    <tr>
        <td>Samantha</td>
        <td>iPod</td>
        <td>Best Buy</td>
        <td>June 3, 2016</td>
    </tr>
    <tr>
        <td>Timothy</td>
        <td>Chromebook</td>
        <td>Amazon</td>
        <td>July 8, 2016</td>
    </tr>
    <tr>
        <td>Jason</td>
        <td>Surface Pro</td>
        <td>Target</td>
        <td>October 2, 2016</td>
    </tr>
</table>

In `purchases`, each tuple represents the relationship between the `name`, `product`, `retailer`, and `date purchased` attributes. 

A relation's *schema* contains its column names, data types, and constraints. For example, the schema of the `purchases` table states that the columns are `name`, `product`, `retailer`, and `date purchased`; it also states that each column contains text.

The following `prices` relation shows the price of certain gadgets at a few retail stores:

<header style="text-align:center"><strong>prices</strong></header>
<table border="1" class="DataFrame">
    <thead>
        <tr>
            <td><b>retailer</b></td>
            <td><b>product</b></td>
            <td><b>price</b></td>
        </tr>
    </thead>
    <tr>
        <td>Best Buy</td>
        <td>Galaxy S9</td>
        <td>719.00</td>
    </tr>
    <tr>
        <td>Best Buy</td>
        <td>iPod</td>
        <td>200.00</td>
    </tr>
    <tr>
        <td>Amazon</td>
        <td>iPad</td>
        <td>450.00</td>
    </tr>
    <tr>
        <td>Amazon</td>
        <td>Battery pack</td>
        <td>24.87</td>
    </tr>
    <tr>
        <td>Amazon</td>
        <td>Chromebook</td>
        <td>249.99</td>
    </tr>
    <tr>
        <td>Target</td>
        <td>iPod</td>
        <td>215.00</td>
    </tr>
    <tr>
        <td>Target</td>
        <td>Surface Pro</td>
        <td>799.00</td>
    </tr>
    <tr>
        <td>Target</td>
        <td>Google Pixel 2</td>
        <td>659.00</td>
    </tr>
    <tr>
        <td>Walmart</td>
        <td>Chromebook</td>
        <td>238.79</td>
    </tr>
</table>

We can then reference both tables simultaneously to determine how much Samantha, Timothy, and Jason paid for their respective gadgets (assuming prices at each store stay constant over time). Together, the two tables form a **relational database**, which is a collection of one or more relations.
The schema of the entire database is the set of schemas of the individual relations in the database.

## Relational Database Management Systems

A relational database can be simply described as a set of tables containing rows of individual data entries. A relational database management system (RDBMSs) provides an interface to a relational database. [Oracle](https://www.wikiwand.com/en/Oracle_Database), [MySQL](https://www.wikiwand.com/en/MySQL), and [PostgreSQL](https://www.wikiwand.com/en/PostgreSQL) are three of the most commonly used RDBMSs used in practice today.

Relational database management systems give users the ability to add, edit, and remove data from databases. These systems provide several key benefits over using a collection of text files to store data, including:

1. Reliable data storage: RDBMSs protect against data corruption from system failures or crashes.
1. Performance: RDBMSs often store data more efficiently than text files and have well-developed algorithms for querying data.
1. Data management: RDBMSs implement access control, preventing unauthorized users from accessing sensitive datasets.
1. Data consistency: RDBMSs can impose constraints on the data entered—for example, that a column `GPA` only contains floats between 0.0 and 4.0.

To work with data stored in a RDBMS, we use the SQL programming language.

## RDBMS vs. pandas

How do RDBMSs and the `pandas` Python package differ? First, `pandas` is not concerned about data storage. Although DataFrames can read and write from multiple data formats, `pandas` does not dictate how the data are actually stored on the underlying computer like a RDBMS does. Second, `pandas` primarily provides methods for manipulating data while RDBMSs handle both data storage and data manipulation, making them more suitable for larger datasets. A typical rule of thumb is to use a RDBMS for datasets larger than several gigabytes. Finally, `pandas` requires knowledge of Python in order to use, whereas RDBMSs require knowledge of SQL. Since SQL is simpler to learn than Python, RDBMSs allow less technical users to store and query data, a handy trait.
