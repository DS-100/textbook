
<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#SQL-Joins" data-toc-modified-id="SQL-Joins-1">SQL Joins</a></span></li><li><span><a href="#Joins" data-toc-modified-id="Joins-2">Joins</a></span><ul class="toc-item"><li><span><a href="#Inner-Join" data-toc-modified-id="Inner-Join-2.1">Inner Join</a></span></li><li><span><a href="#Full/Outer-Join" data-toc-modified-id="Full/Outer-Join-2.2">Full/Outer Join</a></span></li><li><span><a href="#Left-Join" data-toc-modified-id="Left-Join-2.3">Left Join</a></span></li><li><span><a href="#Right-Join" data-toc-modified-id="Right-Join-2.4">Right Join</a></span></li><li><span><a href="#Implicit-Inner-Joins" data-toc-modified-id="Implicit-Inner-Joins-2.5">Implicit Inner Joins</a></span></li></ul></li><li><span><a href="#Joining-Multiple-Tables" data-toc-modified-id="Joining-Multiple-Tables-3">Joining Multiple Tables</a></span></li><li><span><a href="#Summary" data-toc-modified-id="Summary-4">Summary</a></span></li></ul></div>


```python
# HIDDEN
import sqlalchemy
import pandas as pd

# Delete the database if it already exists.
from pathlib import Path
dbfile = Path("sql_joins.db")
if dbfile.exists():
    dbfile.unlink()

sqlite_uri = "sqlite:///sql_joins.db"
sqlite_engine = sqlalchemy.create_engine(sqlite_uri)
```


```python
# HIDDEN
# Make names table
sql_expr = """
CREATE TABLE names(
    cat_id INTEGER PRIMARY KEY,
    name TEXT);
"""
result = sqlite_engine.execute(sql_expr)

# Populate names table
sql_expr = """
INSERT INTO names VALUES 
(0, "Apricot"),
(1, "Boots"),
(2, "Cally"),
(4, "Eugene");
"""

result = sqlite_engine.execute(sql_expr)
```


```python
# HIDDEN
# Make colors table
sql_expr = """
CREATE TABLE colors(
    cat_id INTEGER PRIMARY KEY,
    color TEXT);
"""
result = sqlite_engine.execute(sql_expr)

# Populate colors table
sql_expr = """
INSERT INTO colors VALUES 
(0, "orange"),
(1, "black"),
(2, "calico"),
(3, "white");
"""

result = sqlite_engine.execute(sql_expr)
```


```python
# HIDDEN
# Make ages table
sql_expr = """
CREATE TABLE ages(
    cat_id INTEGER PRIMARY KEY,
    age INT);
"""
result = sqlite_engine.execute(sql_expr)

# Populate ages table
sql_expr = """
INSERT INTO ages VALUES 
(0, 4),
(1, 3),
(2, 9),
(4, 20);
"""

result = sqlite_engine.execute(sql_expr)
```

## SQL Joins

In `pandas` we use the `pd.merge` method to join two tables using matching values in their columns. For example:

```python
pd.merge(table1, table2, on='commmon_column')
```

In this section, we introduce SQL joins. SQL joins are used to combine multiple tables in a relational database.

Suppose we are cat store owners with a database for the cats we have in our store. We have **two** different tables: `names` and `colors`. The `names` table contains the columns `cat_id`, a unique number assigned to each cat, and `name`, the name for the cat. The `colors` table contains the columns `cat_id` and `color`, the color of each cat.

Note that there are some missing rows from both tables - a row with `cat_id` 3 is missing from the `names` table, and a row with `cat_id` 4 is missing from the `colors` table.

<div style="display: flex">
  <div style="margin-right: 2em">
    <header style="text-align:center"><strong>names</strong></header>
    <table>
      <thead>
        <tr>
          <th>cat_id</th>
          <th>name</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>0</td>
          <td>Apricot</td>
        </tr>
        <tr>
          <td>1</td>
          <td>Boots</td>
        </tr>
        <tr>
          <td>2</td>
          <td>Cally</td>
        </tr>
        <tr>
          <td>4</td>
          <td>Eugene</td>
        </tr>
      </tbody>
    </table>
  </div>
  <div>
    <header style="text-align:center"><strong>colors</strong></header>
    <table>
      <thead>
        <tr>
          <th>cat_id</th>
          <th>color</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>0</td>
          <td>orange</td>
        </tr>
        <tr>
          <td>1</td>
          <td>black</td>
        </tr>
        <tr>
          <td>2</td>
          <td>calico</td>
        </tr>
        <tr>
          <td>3</td>
          <td>white</td>
        </tr>
      </tbody>
    </table>
  </div>
</div>

To compute the color of the cat named Apricot, we have to use information in both tables. We can *join* the tables on the `cat_id` column, creating a new table with both `name` and `color`.

## Joins

A join combines tables by matching values in their columns.

There are four main types of joins: inner joins, outer joins, left joins, and right joins. Although all four combine tables, each one treats non-matching values differently.

### Inner Join

<b>Definition:</b> In an inner join, the final table only contains rows that have matching columns in **both** tables.



![Inner Join](https://www.w3schools.com/sql/img_innerjoin.gif)


<b>Example:</b> We would like to join the `names` and `colors` tables together to match each cat with its color. Since both tables contain a `cat_id` column that is the unique identifier for a cat, we can use an inner join on the `cat_id` column.

<b>SQL:</b> To write an inner join in SQL we modify our `FROM` clause to use the following syntax:

```sql
SELECT ...
FROM <TABLE_1>
    INNER JOIN <TABLE_2>
    ON <...>
```

For example:

```sql
SELECT *
FROM names AS N
    INNER JOIN colors AS C
    ON N.cat_id = C.cat_id;
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cat_id</th>
      <th>name</th>
      <th>cat_id</th>
      <th>color</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Apricot</td>
      <td>0</td>
      <td>orange</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Boots</td>
      <td>1</td>
      <td>black</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>Cally</td>
      <td>2</td>
      <td>calico</td>
    </tr>
  </tbody>
</table>

You may verify that each cat name is matched with its color. Notice that the cats with `cat_id` 3 and 4 are not present in our resulting table because the `colors` table doesn't have a row with `cat_id` 4 and the `names` table doesn't have a row with `cat_id` 3. In an inner join, if a row doesn't have a matching value in the other table, the row is not included in the final result.

Assuming we have a DataFrame called `names` and a DataFrame called `colors`, we can conduct an inner join in `pandas` by writing:

```python
pd.merge(names, colors, how='inner', on='cat_id')
```

### Full/Outer Join

<b>Definition:</b> In a full join (sometimes called an outer join), **all records from both tables** are included in the joined table. If a row doesn't have a match in the other table, the missing values are filled in with `NULL`.

![Full outer join](https://www.w3schools.com/sql/img_fulljoin.gif)

<b>Example:</b> As before, we join the `names` and `colors` tables together to match each cat with its color. This time, we want to keep all rows in either table even if there isn't a match.

<b>SQL:</b> To write an outer join in SQL we modify our `FROM` clause to use the following syntax:

```sql
SELECT ...
FROM <TABLE_1>
    FULL JOIN <TABLE_2>
    ON <...>
```

For example:

```sql
SELECT name, color
FROM names N
    FULL JOIN colors C
    ON N.cat_id = C.cat_id;
```

| cat_id        | name          | color    |
| ------------- |---------------|-----------
| 0             | Apricot       | orange   |
| 1             | Boots         | black    |
| 2             | Cally         | calico   |
| 3             | NULL          | white    |
| 4             | Eugene        | NULL     |

Notice that the final output contains the entries with `cat_id` 3 and 4. If a row does not have a match, it is still included in the final output and any missing values are filled in with `NULL`.

In `pandas`:

```
pd.merge(names, colors, how='outer', on='cat_id')
```

### Left Join

<b>Definition:</b> In a left join, all records from the **left table** are included in the joined table. If a row doesn't have a match in the right table, the missing values are filled in with `NULL`.

![left join](https://www.w3schools.com/sql/img_leftjoin.gif)

<b>Example:</b> As before, we join the `names` and `colors` tables together to match each cat with its color. This time, we want to keep all the cat names even if a cat doesn't have a matching color.

<b>SQL:</b> To write an left join in SQL we modify our `FROM` clause to use the following syntax:

```sql
SELECT ...
FROM <TABLE_1>
    LEFT JOIN <TABLE_2>
    ON <...>
```

For example:

```sql
SELECT name, color
FROM names N
    LEFT JOIN colors C
    ON N.cat_id = C.cat_id;
```

| cat_id        | name          | color    |
| ------------- |---------------|-----------
| 0             | Apricot       | orange   |
| 1             | Boots         | black    |
| 2             | Cally         | calico   |
| 4             | Eugene        | NULL     |

Notice that the final output includes all four cat names. Three of the `cat_id`s in the `names` relation had matching `cat_id`s in the `colors` table and one did not (Eugene). The cat name that did not have a matching color has `NULL` as its color.

In `pandas`:

```
pd.merge(names, colors, how='left', on='cat_id')
```

### Right Join

<b>Definition:</b> In a right join, all records from the **right table** are included in the joined table. If a row doesn't have a match in the left table, the missing values are filled in with `NULL`.

![right join](https://www.w3schools.com/sql/img_rightjoin.gif)

<b>Example:</b> As before, we join the `names` and `colors` tables together to match each cat with its color. This time, we want to keep all the cat color even if a cat doesn't have a matching name.

<b>SQL:</b> To write a right join in SQL we modify our `FROM` clause to use the following syntax:

```sql
SELECT ...
FROM <TABLE_1>
    RIGHT JOIN <TABLE_2>
    ON <...>
```

For example:

```sql
SELECT name, color
FROM names N
    RIGHT JOIN colors C
    ON N.cat_id = C.cat_id;
```

| cat_id        | name          | color    |
| ------------- |---------------|-----------
| 0             | Apricot       | orange   |
| 1             | Boots         | black    |
| 2             | Cally         | calico   |
| 3             | NULL          | white    |

This time, observe that the final output includes all four cat colors. Three of the `cat_id`s in the `colors` relation had matching `cat_id`s in the `names` table and one did not (white). The cat color that did not have a matching name has `NULL` as its name.

You may also notice that a right join produces the same result a left join with the table order swapped. That is, `names` left joined with `colors` is the same as `colors` right joined with `names`. Because of this, some SQL engines (such as SQLite) do not support right joins.

In `pandas`:

```
pd.merge(names, colors, how='right', on='cat_id')
```

### Implicit Inner Joins

There are typically multiple ways to accomplish the same task in SQL just as there are multiple ways to accomplish the same task in Python. We point out one other method for writing an inner join that appears in practice called an *implicit join*. Recall that we previously wrote the following to conduct an inner join:

```sql
SELECT *
FROM names AS N
    INNER JOIN colors AS C
    ON N.cat_id = C.cat_id;
```

An implicit inner join has a slightly different syntax. Notice in particular that the `FROM` clause uses a comma to select from two tables and that the query includes a `WHERE` clause to specify the join condition.

```sql
SELECT *
FROM names AS N, colors AS C
WHERE N.cat_id = C.cat_id;
```

When multiple tables are specified in the `FROM` clause, SQL creates a table containing every combination of rows from each table. For example:


```python
sql_expr = """
SELECT *
FROM names N, colors C
"""
pd.read_sql(sql_expr, sqlite_engine)
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
      <th>cat_id</th>
      <th>name</th>
      <th>cat_id</th>
      <th>color</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Apricot</td>
      <td>0</td>
      <td>orange</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>Apricot</td>
      <td>1</td>
      <td>black</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>Apricot</td>
      <td>2</td>
      <td>calico</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>Apricot</td>
      <td>3</td>
      <td>white</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>Boots</td>
      <td>0</td>
      <td>orange</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>Boots</td>
      <td>1</td>
      <td>black</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>Boots</td>
      <td>2</td>
      <td>calico</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>Boots</td>
      <td>3</td>
      <td>white</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2</td>
      <td>Cally</td>
      <td>0</td>
      <td>orange</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2</td>
      <td>Cally</td>
      <td>1</td>
      <td>black</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2</td>
      <td>Cally</td>
      <td>2</td>
      <td>calico</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2</td>
      <td>Cally</td>
      <td>3</td>
      <td>white</td>
    </tr>
    <tr>
      <th>12</th>
      <td>4</td>
      <td>Eugene</td>
      <td>0</td>
      <td>orange</td>
    </tr>
    <tr>
      <th>13</th>
      <td>4</td>
      <td>Eugene</td>
      <td>1</td>
      <td>black</td>
    </tr>
    <tr>
      <th>14</th>
      <td>4</td>
      <td>Eugene</td>
      <td>2</td>
      <td>calico</td>
    </tr>
    <tr>
      <th>15</th>
      <td>4</td>
      <td>Eugene</td>
      <td>3</td>
      <td>white</td>
    </tr>
  </tbody>
</table>
</div>



This operation is often called a *Cartesian product*: each row in the first table is paired with every row in the second table. Notice that many rows contain cat colors that are not matched properly with their names. The additional `WHERE`  clause in the implicit join filters out rows that do not have matching `cat_id` values.

```sql
SELECT *
FROM names AS N, colors AS C
WHERE N.cat_id = C.cat_id;
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cat_id</th>
      <th>name</th>
      <th>cat_id</th>
      <th>color</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Apricot</td>
      <td>0</td>
      <td>orange</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Boots</td>
      <td>1</td>
      <td>black</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>Cally</td>
      <td>2</td>
      <td>calico</td>
    </tr>
  </tbody>
</table>

## Joining Multiple Tables

To join multiple tables, extend the `FROM` clause with additional `JOIN` operators. For example, the following table `ages` includes data about each cat's age.

| cat_id        | age     | 
| ------------- |---------|
| 0             | 4       | 
| 1             | 3       | 
| 2             | 9       | 
| 4             | 20      | 

To conduct an inner join on the `names`, `colors`, and `ages` table, we write:


```python
# Joining three tables

sql_expr = """
SELECT name, color, age
    FROM names n
    INNER JOIN colors c ON n.cat_id = c.cat_id
    INNER JOIN ages a ON n.cat_id = a.cat_id;
"""
pd.read_sql(sql_expr, sqlite_engine)
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
      <th>name</th>
      <th>color</th>
      <th>age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Apricot</td>
      <td>orange</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Boots</td>
      <td>black</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Cally</td>
      <td>calico</td>
      <td>9</td>
    </tr>
  </tbody>
</table>
</div>



## Summary

We have covered the four main types of SQL joins: inner, full, left, and right joins. We use all four joins to combine information in separate relations, and each join differs only in how it handles non-matching rows in the input tables.
