# Preliminary report

The report is limited to two A4 pages. It includes the project description, existing tools, algorithms, pseudocode, two short code excerpts, an actual simulation frame, preliminary measurements, planned experiments, limitations, deliverables and references.

## Before submission

The `\repositoryURL` macro is set to [nidhish1/rl-optimal-control-project](https://github.com/nidhish1/rl-optimal-control-project). The upstream simulator link is a reference, not the project's repository.

Optionally fill `\authorDetails` with your name, student ID and course.

## Compile

Upload the contents of this directory to Overleaf and select `preliminary_report.tex` as the main document (pdfLaTeX). Alternatively:

```bash
cd report
pdflatex -interaction=nonstopmode -halt-on-error preliminary_report.tex
pdflatex -interaction=nonstopmode -halt-on-error preliminary_report.tex
```

Tectonic also works:

```bash
cd report
tectonic preliminary_report.tex
```

The verified PDF is delivered in `../output/pdf/preliminary_report.pdf`. The source ZIP includes the `.tex` file and its image asset, so it can be compiled independently of the project.

The screenshot is copied unchanged from the recorded MPC demonstration. Numerical results come from the saved evaluation/forecast JSON files. Recheck the two-page limit after changing text or adding author details.
