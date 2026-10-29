"""Task-private CPU numerical convention; no shared environment changes."""
import os
os.environ['OPENBLAS_CORETYPE']='Prescott'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
