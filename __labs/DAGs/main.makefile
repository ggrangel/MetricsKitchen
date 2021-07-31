ALL_FIGURE_NAMES=$(shell cat main.figlist)
ALL_FIGURES=$(ALL_FIGURE_NAMES:%=%.pdf)

allimages: $(ALL_FIGURES)
	@echo All images exist now. Use make -B to re-generate them.

FORCEREMAKE:

include $(ALL_FIGURE_NAMES:%=%.dep)

%.dep:
	mkdir -p "$(dir $@)"
	touch "$@" # will be filled later.

main-figure0.pdf: 
	pdflatex -halt-on-error -interaction=batchmode -jobname "main-figure0" "\def\tikzexternalrealjob{main}\input{main}" {; convert -density 300 -transparent white "./image.pdf" "./image.png"}

main-figure0.pdf: main-figure0.md5
