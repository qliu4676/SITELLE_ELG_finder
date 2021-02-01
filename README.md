# SITELLE ELG Finder

Search for Emission-line Galaxy (ELG) candidates in SITELLE datacubes using cross-correlation.

Installation
------------

```bash
  cd <install directory>
  git clone https://github.com/NGC4676/SITELLE_ELG_finder
  cd SITELLE_ELG_finder
  pip install -e .
```

Usage
-----------
1. jupyter notebook/lab
```bash
%run Find_Emission_Candidate.py "A2390C4new.fits" --NAME 'A2390C' -z 0.228 -v --OUT_DIR './output' --DEEP_FRAME "A2390C_deep.fits"
```

2. terminal
```python
python Find_Emission_Candidate.py A2390C4new.fits --NAME A2390C -z 0.228 -v --OUT_DIR ./output --DEEP_FRAME A2390C_deep.fits
```

Three ELG samples are generated:   

A - high S/N Hα+NII candidates   

B - low S/N Hα+NII candidates   

C - (possible) OII+Hβ

When cross-correlation is done, visual inspection is needed by removing dubious objects (delete png) in the directories. Then copy all objects to V and run (e.g. in jupyter notebook):
```python
%run -i Find_Emission_Candidate.py "A2390C4new.fits" --NAME 'A2390C' -w --OUT_DIR './output/'
```

This will generate a list for ELGs with flag=1 Hα+NII candidates and flag=1 for OII+Hβ candidates.

Keyword
-----------

-z: Redshift of the cluster.

--NAME: Name of the cluster (arbitrary string for identification).

-v: Verbose print (default: False).

-p: Verbose plot (default: False).

-w: write ELG list to txt table

--OUT_DIR (optional): Output directory.

--DEEP_FRAME (optional): Path of the deep frame.

--WAVL_MASK (optional): Wavelength masked for sky lines (default: [[7950,8006], [8020,8040], [8230,8280]]).

--sn_thre (optional): S/N threshold for source detection (default: 3).


