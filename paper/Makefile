PAPER = triemap

FMT_FILES = rae.fmt

GHC_OPTS = -dynamic-too
HS_MODULES = Bag RandomType

# -halt-on-error: Halts on first error, rather than prompting the user
REPL_FLAGS := -halt-on-error

all: main

code: $(PAPER).o

$(PAPER).tex: $(PAPER).lhs $(FMT_FILES)
	lhs2TeX --set=main --unset=appendix -o $@ $<

$(PAPER)_ext.tex: $(PAPER).lhs appendix.lhs $(FMT_FILES)
	lhs2TeX --set=main --set=appendix -o $@ $<

appendix.tex: $(PAPER).lhs appendix.lhs $(FMT_FILES)
	lhs2TeX --unset=main --set=appendix -o $@ $<

%.pdf: %.tex refs.bib
	latexmk -pdf $<

# The main paper without appendix
main: $(PAPER).pdf

# Fast, possibly incomplete rebuild. Mostly sufficient for quick rebuilds
fast: $(PAPER).tex
	pdflatex $(REPL_FLAGS) triemap

# Just a standalone appendix.pdf
appendix: appendix.pdf

# Main paper including appendix
extended: $(PAPER)_ext.pdf

# Fast, possibly incomplete rebuild of the appendix. Mostly sufficient for quick rebuilds
appendix_fast: appendix.tex
	pdflatex $(REPL_FLAGS) appendix.tex

$(PAPER).o: $(HS_MODULES:%=%.o)

%.o: %.hs
	ghc $(GHC_OPTS) -c $<

%.hs: %.lhs $(FMT_FILES)
	lhs2TeX --newcode -o $@ $<

watch:
	ls *.fmt *.tex *.lhs | entr make fast

appendix_watch:
	ls *.fmt *.tex *.lhs | entr make appendix_fast

# For submission to arXiv
arxiv.zip: $(PAPER)_ext.tex refs.bib
	@rm -rf arxiv.zip
	@mkdir -p arxiv/
	@# We need to convince arxiv to use pdflatex
	echo "\pdfoutput=1" | cat - $< > arxiv/$<
	bibtex $(basename $<)
	cp $(PAPER)_ext.bbl arxiv/
	cp $(wildcard bench-*) arxiv/
	@# Also arxiv uses hypertex, which doesn't really work for us because we are using cleveref
	echo "nohypertex" > arxiv/00README.XXX
	cd arxiv && zip -r ../arxiv.zip .; cd ..
	@rm -rf arxiv

clean:
	rm -f *.hi *.o *.ptb
	rm -f $(PAPER).pdf $(PAPER).tex $(PAPER).hs
	rm -rf arxiv

.PHONY: all clean code main appendix extended fast appendix_fast watch appendix_watch
.SECONDARY:
