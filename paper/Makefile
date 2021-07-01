PAPER = triemap

FMT_FILES = rae.fmt

GHC_OPTS = -dynamic-too
HS_MODULES = Bag RandomType

all: $(PAPER).pdf

code: $(PAPER).o

%.tex: %.lhs $(FMT_FILES)
	lhs2TeX -o $@ $<

%.pdf: %.tex refs.bib
	latexmk -pdf $<

$(PAPER).o: $(HS_MODULES:%=%.o)

%.o: %.hs
	ghc $(GHC_OPTS) -c $<

%.hs: %.lhs $(FMT_FILES)
	lhs2TeX --newcode -o $@ $<

clean:
	rm -f *.hi *.o *.ptb
	rm -f $(PAPER).pdf $(PAPER).tex $(PAPER).hs

.PHONY: all code
.SECONDARY:
