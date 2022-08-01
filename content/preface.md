# To the Data 100 Student

:::{note}

This book is currently undergoing major updates as we prepare it for
publication. Thanks for your patience as we rearrange and fill in missing
content. We have not removed content from the book entirely. Instead, we have
moved content and marked sections as \[In Progress\] to denote new content that
is actively being worked on.

:::

Data 100 aims to prepare you for real-world data analyses. In theory, drawing
conclusions from data is simple – load a data table, make a plot, and fit a
model. In practice, it is not. Data are messy. Data sources collect data in
different formats. Data values go missing. A simple linear model is not always
appropriate. How do we pick from many possible alternative models? And how do
we generalize our conclusions outside our limited data sample?

The work of a data scientist is to understand and address these questions. To
this end, Data 100 has several key concepts that we revisit throughout the
course.

1. **Data Lifecycle:** The data lifecycle begins with question formulation
   where we take a question of interest and refine it to a question that can be
   answered/studied with data. The data that we plan to use may need to be
   collected or may already be available. Either way, we need to clean,
   explore, and visualize these data before drawing any conclusions or modeling
   our data. Depending on the purpose of our investigation, we often want to
   generalize our findings beyond our data. In short, the life cycle involves
   roughly five steps: (A) Question formulation, (B) Data collection and
   cleaning, (C) Exploratatory data analysis and visualization, (D) Modeling,
   and (E) Generalizing and reporting findings.
1. **From Raw Data to Analyzable Data:** Data scientists must develop
   proficiency in working with data in a variety of forms. This course
   introduces tools and techniques commonly required for real-world data
   analyses, including the `pandas` Python package, data visualization, regular
   expressions, one-hot encodings, data transformations, and querying data from
   databases.
1. **Loss and Estimation:** Data 100 uses model loss as a general
   framework for model fitting. Rather than derive a separate analytic
   solution for each model that we introduce, we rely on gradient descent, an
   elegant optimization-based approach that works well in practice and can
   applied to many types of models.
1. **Descriptive, Predictive, and Inferential Modeling:** Given a single
   dataset, we can pose questions that fall into several categories.
   _Descriptive_ questions ask about the dataset itself – given a dataset about
   a subway system, which lines are most frequently used? _Predictive_ questions
   ask about future data values – next Sunday, how many trains are needed to
   meet expected demand? And _inferential_ questions ask about uncertainty in
   predictions – how many trains should we hold in reserve, in case demand
   spikes more than expected? A data scientist needs to distinguish between
   these types of questions and understand what questions can and cannot be
   answered with the data at hand.

Chapters 1-13 focus on the first two concepts, with a light introduction to the
second two. Chapters 14-27 focus on the second two concepts. This book is
further divided into parts, and each part can be read on its own with minimal
reliance on the other parts.
