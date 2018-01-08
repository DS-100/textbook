.PHONY: help notebooks serve deploy

BLUE=\033[0;34m
NOCOLOR=\033[0m

BOOK_URL=https://ds100.gitbooks.io/principles-and-techniques-of-data-science/content/

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

notebooks:
	@echo "${BLUE}Converting notebooks to HTML.${NOCOLOR}"
	@echo "${BLUE}=============================${NOCOLOR}"

	python convert_notebooks_to_html_partial.py

	@echo ""
	@echo "${BLUE}    Done, output is in notebooks-html${NOCOLOR}"

serve: notebooks
	gitbook install
	gitbook serve

deploy:
ifneq ($(shell git for-each-ref --format='%(upstream:short)' $(shell git symbolic-ref -q HEAD)),origin/gh-pages)
	@echo "Please check out the deployment branch, gh-pages, if you want to deploy your revisions."
	@echo "You might also need to bring the deployment branch up to date by merging it with the staging branch."
	@echo "For example: 'git checkout gh-pages && git pull && git merge staging && make deploy'"
	@echo "(Current branch: $(shell git for-each-ref --format='%(upstream:short)' $(shell git symbolic-ref -q HEAD)))"
	exit 1
endif
	git pull
	make notebooks
	git add -A
	git commit -m "Build notebooks"
	@echo "${BLUE}Deploying book to Gitbook.${NOCOLOR}"
	@echo "${BLUE}=========================${NOCOLOR}"
	git push origin gh-pages
	@echo ""
	@echo "${BLUE}    Done, see book at ${BOOK_URL}.${NOCOLOR}"
