
# Relational Databases and SQL

Thus far we have worked with datasets that are stored as text files on a
computer. While useful for analysis of small datasets, using text files to
store data presents challenges for many real-world use cases.

Many datasets are collected by multiple people—a team of data scientists, for
example. If the data are stored in text files, however, the team will likely
have to send and download new versions of the files each time the data are
updated. Text files alone do not provide a consistent point of data retrieval
for multiple analysts to use. This issue, among others, makes text files
difficult to use for larger datasets or teams.

We often turn to relational database management systems (RDBMSs) to store data,
such as MySQL or PostgreSQL. To work with these systems, we use a query
language called SQL instead of Python. In this chapter, we discuss the
relational database model and introduce SQL.

