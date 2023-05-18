PIP := $(shell which pip)

SPHINXOPTS         ?=
SPHINXBUILD        ?= $(shell which sphinx-build)
SPHINXMULTIVERSION ?= $(shell which sphinx-multiversion)
SOURCEDIR          ?= $(shell readlink -f ${CURDIR})
BUILDDIR           ?= $(shell readlink -f ${CURDIR}/../build)

DIAGRAMS_MK := ${SOURCEDIR}/diagrams.mk
DIAGRAMS    := $(MAKE) -f "${DIAGRAMS_MK}" SOURCE=${SOURCEDIR}
GRAPHVIZ_MK := ${SOURCEDIR}/graphviz.mk
GRAPHVIZ    := $(MAKE) -f "${GRAPHVIZ_MK}" SOURCE=${SOURCEDIR}
DEMOS_MK    := ${SOURCEDIR}/demos.mk
DEMOS       := $(MAKE) -f "${DEMOS_MK}" SOURCE=${SOURCEDIR}
NOTEBOOK_MK := ${SOURCEDIR}/notebook.mk
NOTEBOOK    := $(MAKE) -f "${NOTEBOOK_MK}" SOURCE=${SOURCEDIR}

_CURRENT_PATH := ${PATH}
_PROJ_DIR     := $(shell readlink -f ${SOURCEDIR}/../..)
_LIBS_DIR     := $(shell readlink -f ${SOURCEDIR}/_libs)
_SHIMS_DIR    := $(shell readlink -f ${SOURCEDIR}/_shims)

.EXPORT_ALL_VARIABLES:

PYTHONPATH = ${_PROJ_DIR}:${_LIBS_DIR}
PATH       = ${_SHIMS_DIR}:${_CURRENT_PATH}

.PHONY: all build clean pip

pip:
	@$(PIP) install -r ${_PROJ_DIR}/requirements.txt
	@$(PIP) install -r ${_PROJ_DIR}/requirements-doc.txt

build:
	@$(DIAGRAMS) build
	@$(GRAPHVIZ) build
	@$(DEMOS) build
	@$(NOTEBOOK) build

all: build

clean:
	@$(DIAGRAMS) clean
	@$(GRAPHVIZ) clean
	@$(DEMOS) clean
	@$(NOTEBOOK) clean

cleanplt:
	@$(DEMOS) cleanplt
