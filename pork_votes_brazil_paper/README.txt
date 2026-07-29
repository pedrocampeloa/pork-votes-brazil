When Pork Changes Hands: Coalition Presidentialism, Legislative Capture,
and the Price of Legislative Support in Brazil.

Pedro C. Campelo Albuquerque, Daniel O. Cajueiro, Rafael T. Menezes
University of Brasilia.

Self-contained manuscript package (submission-ready, Public Choice / Springer).

Contents
--------
paper.tex     Manuscript source (LaTeX, sn-jnl class, natbib author-year via sn-basic).
refs.bib      Bibliography (BibTeX).
sn-jnl.cls    Springer Nature journal class file (December 2024 version).
sn-basic.bst  Springer Nature "Basic" author-year bibliography style.
paper.pdf     Compiled manuscript (30 pages).
figs/         Five PDF figures referenced by paper.tex.

Compile
-------
Any modern TeX distribution (TeX Live 2024+, MacTeX, MikTeX) or tectonic
should build the paper without external dependencies. Recommended:

    tectonic paper.tex

or

    pdflatex paper.tex && bibtex paper && pdflatex paper.tex && pdflatex paper.tex

The compilation writes paper.pdf next to paper.tex.

Replication code and data
-------------------------
Code, analysis scripts, and analysis-ready panel data are archived at:

    Zenodo:  https://doi.org/10.5281/zenodo.21378905
    GitHub:  https://github.com/pedrocampeloa/pork-votes-brazil
