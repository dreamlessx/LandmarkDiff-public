# LandmarkDiff MICCAI 2026 Paper — Build System
# Requires: pdflatex, bibtex (from texlive)

MAIN = main
BIB = references

.PHONY: all clean view

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex $(BIB).bib
	pdflatex $(MAIN)
	bibtex $(MAIN)
	pdflatex $(MAIN)
	pdflatex $(MAIN)

clean:
	rm -f $(MAIN).{aux,bbl,blg,log,out,pdf,synctex.gz,fls,fdb_latexmk}

view: $(MAIN).pdf
	xdg-open $(MAIN).pdf 2>/dev/null || open $(MAIN).pdf 2>/dev/null || echo "PDF: $(MAIN).pdf"
