ENVRUN := conda run -p ./env
ENVPYTHON := $(ENVRUN) python

export ENVRUN
export ENVPYTHON

AIWRITING_SAMPLE_SIZE ?= 0

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
	$(ENVPYTHON) ./src/make_paper_jsons.py --papers ./papers --metadata ./papers/metadata.jsonl --output ./workarea/paper_jsons --samples $(AIWRITING_SAMPLE_SIZE)

./workarea/Makefile.experiments: $(wildcard ./src/*.py)
	@test -d ./env || (echo "Create the python environment first" && false)
	@test -d ./workarea/paper_jsons || (echo "Run preprocessing first" && false)
	$(ENVPYTHON) ./src/create_experiment_makefile.py --papers ./workarea/paper_jsons --output ./workarea/Makefile.experiments

expmkf: ./workarea/Makefile.experiments

original_pangram: ./workarea/Makefile.experiments
	$(MAKE) -f ./workarea/Makefile.experiments original_pangram
