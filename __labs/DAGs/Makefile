all: compile clean move

compile:
	@rm -f *.md5 *.auxlock *.aux *.bbl *.blg *.idx *.ind *.lof *.lot *.out *.toc *.acn *.acr *.alg *.glg *.glo *.gls *.ist *.fls *.log *.fdb_latexmk *.nlo *.ilg *.nls *.nlg *.bcf *.lol *.run.xml *.synctex.gz .log
	@find . -type f -name '*.aux' -delete
	@echo "compiling document..."
	pdflatex -synctex=1 -interaction=nonstopmode -file-line-error $(filename).tex
	# biber $(filename)
	# makeindex $(filename)
	pdflatex -synctex=1 -interaction=nonstopmode -file-line-error $(filename).tex
	pdflatex -synctex=1 -interaction=nonstopmode -file-line-error $(filename).tex
	# pdflatex -synctex=1 -interaction=nonstopmode -file-line-error $(filename).tex
	@echo "document has been successfully generated!"

clean:
	@echo -n "removing auxiliaries files...\n"
	@rm -f *.md5 *.auxlock *.aux *.bbl *.blg *.idx *.ind *.lof *.lot *.out *.toc *.acn *.acr *.alg *.glg *.glo *.gls *.ist *.fls *.log *.fdb_latexmk *.nlo *.ilg *.nls *.nlg *.bcf *.lol *.run.xml *.synctex.gz .log
	@find . -type f -name '*.aux' -delete
	@echo "finish!"

move:
	@cp main.pdf /home/rangelgbr/Workspace/Masters/02_Econometria\ do\ Setor\ Público/Listas/Prova/dag.pdf
