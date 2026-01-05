ENVRUN := conda run -p ./env
ENVPYTHON := $(ENVRUN) python



all:
	@echo "Do not run this target for now"
	@echo "Available targets: env, preprocess"

env:
	rm -rf ./env
	conda create --yes --prefix ./env python=3.12 pip
	$(ENVRUN) pip install -r ./requirements.txt

preprocess:
	@test -d ./env || (echo "Create the python environment first" && false)
	mkdir -p ./workarea/paper_jsons
	$(ENVPYTHON) ./src/make_paper_jsons.py --papers ./papers --metadata ./papers/metadata.jsonl --output ./workarea/paper_jsons
