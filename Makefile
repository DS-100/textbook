# Generate chapter list
CHAPTERS=$(shell ls -1 notebooks | grep -E ^\\d+$)

.PHONY: help build notebooks serve deploy pdf $(CHAPTERS)

BLUE=\033[0;34m
NOCOLOR=\033[0m

VERSION=v1

BOOK_URL=https://ds100.gitbooks.io/textbook/content/
LIVE_URL=https://ds100.gitbooks.io/textbook/content/v/$(VERSION)

BINDER_REGEXP=.*"message": "([^"]+)".*

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

notebooks: ## Convert notebooks to HTML pages
	@echo "${BLUE}Converting notebooks to HTML.${NOCOLOR}"
	@echo "${BLUE}=============================${NOCOLOR}"
	rm -rf ch/*
	@echo "${BLUE}Removed existing HTML files.${NOCOLOR}"
	@echo ""
	python scripts/convert_notebooks_to_html_partial.py
	@echo "${BLUE}Done, output is in notebooks-html${NOCOLOR}"
	@echo ""

pdf: ## Generates PDF of textbook
	python scripts/create_single_page_html.py
	make website
	ebook-convert _site/book.html book.pdf --chapter "//h:h1" --level1-toc "//h:h1" --level2-toc "//h:h2"

chNN: ## Converts a specific chapter's notebooks (e.g. make 02)
	@echo To use this command, replace NN with the chapter number. Example:
	@echo "  make 01"

$(CHAPTERS): ## Converts a specific chapter's notebooks (e.g. make ch02)
	rm -rf ch/$@/*
	python scripts/convert_notebooks_to_html_partial.py notebooks/$@/*.ipynb

website:
	jekyll build

build: ## Run build steps
	make notebooks website pdf

serve: build ## Run Jekyll to preview changes locally
	jekyll serve


clean: ## Removes generated files (Jekyll and notebook conversion output)
	find notebooks -name '*.md' -exec rm {} \;
	find notebooks -type d -name '*_files' -exec rm -rf {} \;
	rm ch/*/*.html
	rm notebooks-images/*
	jekyll clean

deploy: ## Publish textbook
ifneq ($(shell git for-each-ref --format='%(upstream:short)' $(shell git symbolic-ref -q HEAD)),origin/master)
	@echo "Please check out the deployment branch, master, if you want to deploy your revisions."
	@echo "For example: 'git checkout master && make deploy'"
	@echo "(Current branch: $(shell git for-each-ref --format='%(upstream:short)' $(shell git symbolic-ref -q HEAD)))"
	exit 1
endif
	git pull
	make build
	git add -A
	git commit -m "Build textbook"
	@echo "${BLUE}Deploying book to Gitbook.${NOCOLOR}"
	@echo "${BLUE}=========================${NOCOLOR}"
	git push origin master
	@echo ""
	@echo "${BLUE}Done, see book at ${BOOK_URL}.${NOCOLOR}"
	@echo "${BLUE}Updating Binder image in background (you will see${NOCOLOR}"
	@echo "${BLUE}JSON output in your terminal once built).${NOCOLOR}"
	make ping_binder

ping_binder: ## Force-updates BinderHub image
	curl -s https://mybinder.org/build/gh/DS-100/textbook/master |\
		grep -E '${BINDER_REGEXP}' |\
		sed -E 's/${BINDER_REGEXP}/\1/' &
