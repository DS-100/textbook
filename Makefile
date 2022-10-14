server.PHONY: help build clean

CONTENT = content/_config.yml content/_toc.yml content/*.md content/ch

HTML_DIR = content/_build/html

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

build: ## Builds book html
	jupyter-book build content

rebuild: clean ## Forces complete rebuild of book html
	$(MAKE) build

# Install fswatch first: https://github.com/emcrisostomo/fswatch
watch: ## Rebuilds book when files change (needs fswatch installed)
	@echo Watching content/ch for changes...
	fswatch -0 $(CONTENT) --one-per-batch | xargs -0 -n 1 -I {} $(MAKE) build

server: ## Starts python server that serves html
	mkdir -p $(HTML_DIR)
	cd $(HTML_DIR) && python -m http.server  --bind 127.0.0.1 8000

serve: watch server ## Use -j2 flag to watch and serve content with one command

clean:
	jupyter-book clean content

stage:
	git push test sam-reordering:master
