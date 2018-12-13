
# Exploratory Data Analysis

In exploratory data analysis (EDA), the third major component of the data science lifecycle, we summarize, visualize, and transform data in order to understand them more deeply. John Tukey, the statistician that defined the term EDA, writes:

> ‘Exploratory data analysis’ is an attitude, a state of flexibility, a willingness to look for those things that we believe are not there, as well as those we believe to be there.

A student of data science may find this definition unsatisfying — will an attitude alone generate a data analysis? Tukey's point, however, is that we should understand the data before rushing to apply statistical tests. We would benefit from remembering his words in today's resurgence of algorithmic decision-making.

Still, data scientists need to understand their tools for conducting data exploration. Each tool, whether computational or statistical, comes with limitations. For example, we have seen that `pandas` exclusively works with data in tabular format. In this chapter, we begin with general-purpose tools with few constraints and conclude with powerful tools that have many constraints.

We frame exploratory data analysis in the process of using tools to discovering precise constraints our data satisfy. We demonstrate the discovery of the following constraints:

1. Data files can be read into memory as tables.
2. Data columns have uniform data types, and data rows have consistent granularity.
3. Numeric data variables have a known center, spread, and distribution.
4. Relationships between data variables are known.

We do not believe our focus on "constraints" is at odds with Tukey's focus on "flexibility". As we will see, data often violate desired constraints, and these violations have in fact motivated many new tools and techniques for working with data.
