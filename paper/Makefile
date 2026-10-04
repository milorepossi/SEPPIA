# latexmk on this machine is missing Unicode::Normalize, so drive pdflatex
# directly. Three passes: refs, then bibtex, then resolve citations.
PY ?= python3

all: main.pdf

main.pdf: main.tex tables.tex numbers.tex refs.bib figures/ladder_boltz_pca20.png
	$(PY) make_tables.py
	pdflatex -interaction=nonstopmode main.tex
	bibtex main
	pdflatex -interaction=nonstopmode main.tex
	pdflatex -interaction=nonstopmode main.tex
	@! grep -qE "^! " main.log || { echo "LaTeX error, see main.log"; exit 1; }
	@! grep -qiE "Warning: (Citation|Reference).*undefined" main.log \
		|| { echo "undefined citation or reference"; exit 1; }
	@echo "built main.pdf"

# Verify every quoted number still matches RESULTS/boltz/*/metrics.json.
check:
	$(PY) make_tables.py --check

clean:
	rm -f main.aux main.bbl main.blg main.log main.out main.pdf b[0-9].log build.log

.PHONY: all check clean
