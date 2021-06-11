PAPER = triemap

FMT_FILES = rae.fmt

all: $(PAPER).pdf

%.tex: %.lhs $(FMT_FILES)
	lhs2TeX -o $@ $<

%.pdf: %.tex refs.bib
	pdflatex $<

.PHONY: all
