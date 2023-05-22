# Preface

Data science is exciting work. The ability to draw conclusions from messy data
is valuable for all kinds of decisions across business, medicine, policy, and
more. This book, _Learning Data Science_, aims to prepare readers to do data
science. To achieve this, we've written this book to contain the following
special features:

1. **A focus on the fundamentals.** Technologies come and go. We talk in detail
   about individual technologies in this book, but our goal is to equip readers
   with the fundamental building blocks of data science. We do this by teaching
   readers how to think about data science problems and challenges, which will
   serve readers even as technologies change.
1. **Covering the entire data science lifecycle.** Instead of just focusing on
   a single topic, like working with data tables or machine learning, we cover
   the entire data science lifecycle -- the process of asking a question,
   obtaining data, understanding the data, and understanding the world.
1. **Only using real data.** To be prepared for real data analyses, we consider
   it essential to see examples that use real data. We chose all of the dataset
   examples presented in this book by carefully picking from real-world data
   analyses that have made an impact, not hypothetical or generated data.
1. **Applying concepts through case studies.** We've sprinkled in extended case
   studies in the book that replicate or extend analyses from other
   data scientists. These case studies show readers how to apply concepts in
   real settings.
1. **Both computational and inferential thinking.** On the job, data scientists
   need to foresee how the decisions they make when writing code might affect
   statistical conclusions. To prepare readers for their future work, _Learning
   Data Science_ integrates both computational and statistical thinking. We
   also motivate statistical concepts through simulations rather than
   mathematical proofs.

The text and code for this book are open source and [available on
GitHub][github].

[github]: https://github.com/DS-100/textbook/

## Expected Background Knowledge

We expect that readers are proficient in Python and understand how to use
built-in data structures like lists, dictionaries, and sets; import and use
functions and classes from other packages; and write functions from scratch. We
will use the `numpy` Python package without introduction but don't expect
readers to have much prior experience using it.

Readers will get more from this book if they also know a bit of probability,
calculus, and linear algebra, but we aim to explain mathematical ideas
intuitively.

## Organization of the Book

This book has 21 chapters, divided into six parts:

- **Part 1 (Ch 1-5): The Data Science Lifecycle** describes what the
  lifecycle is, makes one full pass through the lifecycle at a basic level, and
  introduces terminology that we use throughout the book. The part concludes
  with a short case study about bus arrival times.
- **Part 2 (Ch 6-7): Rectangular Data** introduces data frames and
  relations and how to wrote code to manipulate data using `pandas` and SQL.
- **Part 3 (Ch 8-12): Understanding the Data** is all about obtaining data,
  discovering its traits, and spotting issues. After understanding these
  concepts, a reader can take a data file and explain all of the data's
  interesting features to someone else. This part ends with a case study
  about air quality.
- **Part 4 (Ch 13-14): Other Data Sources** looks at alternative sources of
  data like text, binary, and data from the Web which all have important
  roles in data science.
- **Part 5 (Ch 15-18): Linear Modeling** focuses on the understanding the world
  using data. It covers traditional topics like confidence intervals and
  hypothesis testing in additional to modeling, prediction, and inference.
  This part ends with a case study about predicting donkey weights for
  veterinarians in Kenya.
- **Part 6 (Ch 19-21): Classification** completes our study or supervised
  learning through logistic regression and optimization. It ends with a case
  study about predicting whether news articles make real or fake statements.
