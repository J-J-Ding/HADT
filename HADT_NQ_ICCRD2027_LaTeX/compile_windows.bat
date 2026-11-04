@echo off
setlocal
pdflatex -interaction=nonstopmode -halt-on-error main.tex
if errorlevel 1 exit /b 1
bibtex main
if errorlevel 1 exit /b 1
pdflatex -interaction=nonstopmode -halt-on-error main.tex
if errorlevel 1 exit /b 1
pdflatex -interaction=nonstopmode -halt-on-error main.tex
if errorlevel 1 exit /b 1
echo Build completed: main.pdf
