.PHONY: docs test unittest resource

PYTHON := $(shell which python)

PROJ_DIR      := .
DOC_DIR       := ${PROJ_DIR}/docs
BUILD_DIR     := ${PROJ_DIR}/build
DIST_DIR      := ${PROJ_DIR}/dist
TEST_DIR      := ${PROJ_DIR}/test
TESTFILE_DIR  := ${TEST_DIR}/testfile
DATASET_DIR   := ${TESTFILE_DIR}/dataset
SRC_DIR       := ${PROJ_DIR}/imgutils
TEMPLATES_DIR := ${PROJ_DIR}/templates
RESOURCE_DIR  := ${PROJ_DIR}/resource

RANGE_DIR      ?= .
RANGE_TEST_DIR := ${TEST_DIR}/${RANGE_DIR}
RANGE_SRC_DIR  := ${SRC_DIR}/${RANGE_DIR}

GAMES ?= arknights fgo genshin girlsfrontline azurlane

COV_TYPES ?= xml term-missing

package:
	$(PYTHON) -m build --sdist --wheel --outdir ${DIST_DIR}
clean:
	rm -rf ${DIST_DIR} ${BUILD_DIR} *.egg-info \
		$(shell find ${SRC_DIR}/games -name index.json -type f) \
		$(shell find ${SRC_DIR}/games -name danbooru_tags.json -type f) \
		$(shell find ${SRC_DIR}/games -name pixiv_names.json -type f) \
		$(shell find ${SRC_DIR}/games -name pixiv_characters.json -type f) \
		$(shell find ${SRC_DIR}/games -name pixiv_alias.yaml -type f)

test: unittest

unittest:
	UNITTEST=1 \
		pytest "${RANGE_TEST_DIR}" \
		-sv -m unittest \
		$(shell for type in ${COV_TYPES}; do echo "--cov-report=$$type"; done) \
		--cov="${RANGE_SRC_DIR}" \
		$(if ${MIN_COVERAGE},--cov-fail-under=${MIN_COVERAGE},) \
		$(if ${WORKERS},-n ${WORKERS},)

docs:
	$(MAKE) -C "${DOC_DIR}" build
pdocs:
	$(MAKE) -C "${DOC_DIR}" prod

dataset:
	mkdir -p ${DATASET_DIR}
	if [ ! -d ${DATASET_DIR}/chafen_arknights ]; then \
		hfutils download -r deepghs/chafen_arknights -t dataset -a dataset.zip -o ${DATASET_DIR}/chafen_arknights; \
	fi
	if [ ! -d ${DATASET_DIR}/monochrome_danbooru ]; then \
		hfutils download -r deepghs/monochrome_danbooru -t dataset -a dataset.zip -o ${DATASET_DIR}/monochrome_danbooru; \
	fi
	if [ ! -d ${DATASET_DIR}/images_test_v1 ]; then \
		hfutils download -r deepghs/character_similarity -t dataset -a images_test_v1.tar.xz -o ${DATASET_DIR}/images_test_v1; \
	fi
	if [ ! -d ${DATASET_DIR}/unsplash_1000 ]; then \
		hfutils download -r deepghs/realutils_unittest -a unsplash_1000.zip -o ${DATASET_DIR}/unsplash_1000; \
	fi
