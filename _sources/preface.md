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

We've also included notes at the end of the book with resources to learn more
about many of the topics this book introduces.

## Conventions Used in This Book

The following typographical conventions are used in this book:

_Italic_:: Indicates new terms, URLs, email addresses, filenames, and file
extensions.

`Constant width`:: Used for program listings, as well as within paragraphs to
refer to program elements such as variable or function names, databases, data
types, environment variables, statements, and keywords.

**`Constant width bold`**:: Shows commands or other text that should be typed
literally by the user.

_`Constant width italic`_:: Shows text that should be replaced with
user-supplied values or by values determined by context.

```{admonition}
This element signifies a tip or suggestion.
```

```{note}
This element signifies a general note.
```

```{warning}
This element indicates a warning or caution.
```

### Using Code Examples

Supplemental material (code examples, exercises, etc.) is available for
download at link: https://learningds.org.

If you have a technical question or a problem using the code examples, please
send email to bookquestions@oreilly.com.

This book is here to help you get your job done. In general, if example code is
offered with this book, you may use it in your programs and documentation. You
do not need to contact us for permission unless you’re reproducing a
significant portion of the code. For example, writing a program that uses
several chunks of code from this book does not require permission. Selling or
distributing examples from O’Reilly books does require permission. Answering a
question by citing this book and quoting example code does not require
permission. Incorporating a significant amount of example code from this book
into your product’s documentation does require permission.

We appreciate, but generally do not require, attribution. An attribution
usually includes the title, author, publisher, and ISBN. For example: "Learning
Data Science by Sam Lau, Deborah Nolan, and Joey Gonzalez (O'Reilly). Copyright
2023 Sam Lau, Deborah Nolan, and Joey Gonzalez, 978-0-596-xxxx-x."

If you feel your use of code examples falls outside fair use or the permission
given above, feel free to contact us at bookquestions@oreilly.com.

### O'Reilly Online Learning

```{note}
For more than 40 years, [O'Reilly Media](https://oreilly.com)
has provided technology and business training, knowledge, and insight to
help companies succeed.
```

Our unique network of experts and innovators share their knowledge and
expertise through books, articles, and our online learning platform. O’Reilly’s
online learning platform gives you on-demand access to live training courses,
in-depth learning paths, interactive coding environments, and a vast collection
of text and video from O'Reilly and 200+ other publishers. For more
information, visit [https://oreilly.com](https://oreilly.com).

### How to Contact Us

Please address comments and questions concerning this book to the publisher:

<ul class="simplelist">
  <li>O’Reilly Media, Inc.</li>
  <li>1005 Gravenstein Highway North</li>
  <li>Sebastopol, CA 95472</li>
  <li>800-998-9938 (in the United States or Canada)</li>
  <li>707-829-0515 (international or local)</li>
  <li>707-829-0104 (fax)</li>
</ul>

We have a web page for this book, where we list errata, examples, and any
additional information. You can access this page at
link: https://learningds.org.

Email bookquestions@oreilly.com to comment or ask technical questions about
this book.

For news and information about our books and courses, visit https://oreilly.com.

Find us on Facebook: https://facebook.com/oreilly

Follow us on Twitter: https://twitter.com/oreillymedia

Watch us on YouTube: https://www.youtube.com/oreillymedia

## Acknowledgements

We are incredibly thankful for the O'Reilly team for all their work to publish
this book, especially Melissa Potter, Jess Haberman, Aaron Black, Danny
Elfanbaum, and Mike Loukides. This book is based on six years of our joint
experience teaching Principles and Techniques of Data Science, an undergraduate
course at the University of California, Berkeley. We've benefited from
co-teaching with other with other instructors, and we especially want to
mention Joe Hellerstein for insights around data wrangling, Fernando Perez for
NetCDF, and Josh Hug for the inspiration for the PurpleAir case study.

We thank the people who were kind enough to share their data with us for
the writing of this book:

- Google Flu Trends?
- Wikipedia data?
- Stan Lipovetsky

We also make use of many publicly available datasets in this book, and we thank
the people and organizations that made this data available:

- CalEnviroScreen Project
- National Oceanic and Atmospheric Administration (NOAA)
- US Air Quality System
- PurpleAir
- Jake VanderPlas
- Washington State Transportation Center
- American Kennel Club
- US Social Security Department
- U.S. Substance Abuse and Medical Health Services Administration (SAMHSA)
- DataSF
- San Francisco Chronicle
- Cherry Blossom Run
- US Bureau of Labor Statistics
- FiveThirtyEight
- American Presidency Project
- Climate Data Store
- Spotify API
- Wikipedia
- Opportunity Insights and Raj Chetty
- California Department of Fish and Game
- UK Donkey Sanctuary
- Roy Lawrence Rich
- Kai Shu
