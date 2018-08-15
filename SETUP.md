# Setting up the Textbook for Local Development

## How the Textbook Works

The textbook is built using Gitbook ([docs](https://toolchain.gitbook.com/)).
All pages live in the `chNN` folders e.g. `ch12/linear_grad.md`. Some pages are
Markdown files that just contain text content (for example,
`ch12/linear_models.md`).

Most pages, however, are generated from Jupyter notebooks that live in the
`notebooks/` folder. For example, the `notebooks/ch10/modeling_simple.ipynb`
notebook will automatically built into the `ch10/modeling_simple.md` page in
the textbook.

To add a page into the textbook, create a new notebook in the corresponding
chapter in the `notebooks/` folder. Then, run `make build` to generate the
Markdown file. Finally, add a link to the Markdown file in `SUMMARY.md` to put
the page in the table of contents for the book.

## Installing Dependencies

To run the textbook locally, you need a recent version of Jupyter Notebook or
Jupyterlab and Python 3.6 or higher. To install these, we suggest using the
Anaconda installer for Python 3.6: https://www.anaconda.com/download .

You will also need to install a recent version of NodeJS (8.0.0 or higher). See
https://nodejs.org/en/ for installation.

Finally, you need to have git installed to clone the repo and make pull
requests: https://git-scm.com/book/en/v2/Getting-Started-Installing-Git .

To check your setup, you may run these commands in your terminal. They should
run without error.

```bash
git --version # Should output a version >= 2.0.0
python --version # Should output a version >= 3.6.0
node --version # Should output a version >= 8.0.0
npm --version # Should output a version >= 5.0.0
```

## Repo Setup

Run the following in your terminal:

```bash
git clone https://github.com/DS-100/textbook # Creates a textbook/ folder
cd textbook
```

Now, install the Gitbook CLI tools.

```bash
npm install gitbook-cli
```

## Running the Textbook Locally

To run the textbook locally, run:

```bash
make serve
```

This starts a local Gitbook server that you can view by visiting
http://localhost:4000/ in your web browser.

If you create or modify a Markdown file, the Gitbook server will automatically
refresh.

If you create or modify a notebook, you will have to run `make build` in order
to regenerate the Markdown files and refresh the Gitbook server.
