all: triemap.pdf

%.pdf: %.tex refs.bib
	pdflatex $<

.PHONY: all
