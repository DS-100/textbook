.PHONY: help build notebooks serve deploy

BLUE=\033[0;34m
NOCOLOR=\033[0m

BOOK_URL=https://ds100.gitbooks.io/textbook/content/

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

notebooks: ## Convert notebooks to HTML pages
	@echo "${BLUE}Converting notebooks to HTML.${NOCOLOR}"
	@echo "${BLUE}=============================${NOCOLOR}"

	python convert_notebooks_to_html_partial.py
	touch SUMMARY.md

	@echo "${BLUE}Done, output is in notebooks-html${NOCOLOR}"
	@echo ""

section_labels: ## Add section labels
	@echo "${BLUE}Adding section labels to SUMMARY.md...${NOCOLOR}"

	python add_section_numbers_to_book.py
	@echo

build: ## Run build steps
	make notebooks section_labels

serve: build ## Run gitbook to preview changes locally
	gitbook install
	gitbook serve

deploy: ## Publish gitbook
ifneq ($(shell git for-each-ref --format='%(upstream:short)' $(shell git symbolic-ref -q HEAD)),origin/gh-pages)
	@echo "Please check out the deployment branch, gh-pages, if you want to deploy your revisions."
	@echo "You might also need to bring the deployment branch up to date by merging it with the staging branch."
	@echo "For example: 'git checkout gh-pages && git pull && git merge staging && make deploy'"
	@echo "(Current branch: $(shell git for-each-ref --format='%(upstream:short)' $(shell git symbolic-ref -q HEAD)))"
	exit 1
endif
	git pull
	make build
	git add -A
	git commit -m "Build textbook"
	@echo "${BLUE}Deploying book to Gitbook.${NOCOLOR}"
	@echo "${BLUE}=========================${NOCOLOR}"
	git push origin gh-pages
	@echo ""
	@echo "${BLUE}Done, see book at ${BOOK_URL}.${NOCOLOR}"
