PAPER = triemap

FMT_FILES = rae.fmt

all: $(PAPER).pdf

code: $(PAPER).o

%.tex: %.lhs $(FMT_FILES)
	lhs2TeX -o $@ $<

%.pdf: %.tex refs.bib
	pdflatex $<

%.o: %.hs
	ghc -c $<

%.hs: %.lhs $(FMT_FILES)
	lhs2TeX --newcode -o $@ $<

clean:
	rm -f *.hi *.o *.ptb
	rm -f $(PAPER).pdf $(PAPER).tex $(PAPER).hs

.PHONY: all code
