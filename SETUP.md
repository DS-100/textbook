# Making changes to the book

To develop the book locally, you first need to set up a Python environment with
all the packages used to build the book. Edit the book by editing the
Jupyter notebooks in the `content/` folder. To publish changes to the live
book, make a pull request on GitHub. This file contains instructions for all of
these steps.

## Python environment setup

Follow these steps to set up the textbook locally. You only have to go through
these steps once per machine.

These instructions were tested for OSX 10.15. We assume that you know how to
run commands on the `bash` command line. We also assume you have the following
command-line tools installed:

- `conda`, the Python package manager ([installation instructions][conda])
- `git`, the version control tool ([installation instructions][git])
- `brew`, the macOS package manager ([installation instructions][brew])

[conda]: https://docs.anaconda.com/anaconda/install/
[git]: https://git-scm.com/book/en/v2/Getting-Started-Installing-Git
[brew]: https://brew.sh/

1. **Download the book files to your computer.** Open a terminal, navigate to a
   folder for the book files, and run:

   ```bash
   git clone git@github.com:DS-100/textbook.git
   cd textbook # Navigates into the book folder
   ```

1. **Create a `conda` environment with the textbook's required packages.** Run
   the following command:

   ```bash
   conda env create -f environment.yml
   ```

   To check that this command succeeds, run:

   ```bash
   conda info --envs
   ```

   And verify that the `textbook` environment appears in the list.

1. **Install fswatch.** This step is optional, but improves development
   workflow. If you follow this step, you can use the `make watch` command to
   automatically rebuild the book when you make changes locally instead of
   running `make build` to manually rebuild the book. Run:

   ```bash
   brew install fswatch
   ```

## Previewing book changes locally

Follow these steps **each time** you begin working on the book.

1. **Navigate to the `textbook/` folder in your terminal.**
1. **Activate the `textbook` Python environment.** Run:

   ```bash
   source activate textbook
   ```

1. **Checkout a `git` branch for your work.** To make book changes easier to
   track for collaborators, we don't make changes to the `master` branch of the
   textbook. Instead, create a new branch by running:

   ```bash
   git branch [branch_name]
   git checkout [branch_name]
   ```

   Replace `[branch_name]` with the name of your branch. For example, if Sam
   wants to create a branch named `sam-decisiontrees`, he would run:

   ```bash
   git branch sam-decisiontrees
   git checkout sam-decisiontrees
   ```

   The `git branch` command creates a new `git` branch. It will fail if the
   branch already exists; skip this command if this is the case. The
   `git checkout` command switches to a branch. It will do nothing if you are
   already on the branch.

   To check that you performed this step successfully, you should see this
   output when you run `git branch`:

   ```bash
   $ git branch
   master
   * sam-decisiontrees
   ```

   You should **not** see this:

   ```bash
   $ git branch
   * master
   sam-decisiontrees
   ```

   This output means that you are still on the `master` branch, not the one you
   created.

1. **Start the book build system.** Run:

   ```bash
   make build
   make -j2 serve
   ```

   This step builds the book once, then starts a process that automatically
   rebuilds the book whenever you change book content. Once this process
   is running, open http://localhost:8000/ to view the book locally.

1. **Start a Jupyter notebook server.** In a new terminal tab or window,
   navigate to the `textbook/` folder and run `source activate textbook` again.
   Then, run:

   ```bash
   jupyter notebook
   ```

   This should open your browser to a Jupyter server that lists the textbook
   files. You should see a `content/` folder which contains all the book's
   content.

1. **Make changes to book content.** Every page of the book is a Jupyter
   notebook within the `content/` folder. To change a page of the book, edit
   the corresponding notebook for that page. Whenever a notebook is saved, the
   terminal window with the `make -j2 serve` command will automatically rebuild
   the book locally, so you can refresh your http://localhost:8000/ browser tab
   to see how the changes will appear in the final book.

   Note: To see the mapping between textbook pages and Jupyter notebooks, see
   the `content/_toc.yml` file. As an aside, saving the `content/_toc.yml` file
   will force a complete rebuild of the book which is convenient when changes
   to a notebook appear not to change the book.

## Submitting your changes for review

1. **Commit your changes locally.** Once you are ready to submit your changes,
   run these commands in your terminal:

   ```bash
   git add -A                            # Stages all changes
   git status                            # Lists all staged changes
   git commit -m '[your commit message]' # Makes a git commit
   ```

   Replace `[your commit message]` with a short (fewer than 72 character)
   description of your changes. For example:

   ```bash
   git commit -m 'Write 19.3 (PCA in practice)'
   ```

1. **Make a pull request.** A GitHub pull request allows a collaborator to
   review and make comments on your changes. Once approved, the collaborator
   can merge the changes into the live book. Run:

   ```bash
   git push origin HEAD # Push current branch to the same branch on GitHub
   ```

   Now, open https://github.com/DS-100/textbook in your browser. You should see
   a green button titled "Compare & pull request". Click that button. Fill out
   the form on the resulting page with a title and description for your
   changes. Finally, click the "Create pull request" button.

   Example pull request: https://github.com/DS-100/textbook/pull/103
