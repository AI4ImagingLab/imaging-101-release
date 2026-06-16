Implementation reference lineage for this cleaned task:

- OLAF / pyolaf wave-optics light-field toolbox family: https://gitlab.lrz.de/IP/olaf/

This task does not vendor the full external package. Instead, the cleaned
benchmark keeps a compact local wrapper around the Broxton-style wave-model
system builder and Richardson-Lucy deconvolution workflow used for the
synthetic validation in `main.py`.
