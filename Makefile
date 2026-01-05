ENVRUN := conda run -p ./env
ENVPYTHON := $(ENVRUN) python

all:
	@echo "Do not run this target for now"
	@echo "Available targets: env"

env:
	rm -rf ./env
	conda create --yes --prefix ./env python=3.12 pip
	$(ENVRUN) pip install -r ./requirements.txt
