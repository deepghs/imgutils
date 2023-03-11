# You can set these variables from the command line, and also
# from the environment for the first two.
PROJ_DIR           ?= ${CURDIR}
SPHINXOPTS         ?=
SPHINXBUILD        ?= $(shell which sphinx-build)
SPHINXMULTIVERSION ?= $(shell which sphinx-multiversion)
SOURCEDIR          ?= ${PROJ_DIR}/source
BUILDDIR           ?= ${PROJ_DIR}/build

# Minimal makefile for Sphinx documentation
ALL_MK := ${SOURCEDIR}/all.mk
ALL    := $(MAKE) -f "${ALL_MK}" SOURCE=${SOURCEDIR}

.EXPORT_ALL_VARIABLES:

NO_CONTENTS_BUILD = true

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
# Put it first so that "make" without argument is like "make help".
.PHONY: help contents build html prod clean sourcedir builddir Makefile

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

contents:
	@$(ALL) build
build: html
html: contents
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@touch "$(BUILDDIR)/html/.nojekyll"
prod:
	@NO_CONTENTS_BUILD='' $(SPHINXMULTIVERSION) "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O)
	@cp main_page.html "$(BUILDDIR)/html/index.html"
	@touch "$(BUILDDIR)/html/.nojekyll"

clean:
	@$(ALL) clean
	@$(SPHINXBUILD) -M clean "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

sourcedir:
	@echo $(shell readlink -f ${SOURCEDIR})
builddir:
	@echo $(shell readlink -f ${BUILDDIR}/html)