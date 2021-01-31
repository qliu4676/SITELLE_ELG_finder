# SITELLE ELG Finder

Search for Emission-line Galaxy (ELG) candidates in the SITELLE datacube.

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

Keyword
-----------

-z: Redshift of the cluster.

-v: Verbose print.

-v: Verbose plot.

--NAME: Name of the cluster (arbitrary string for identification).

--OUT_DIR (optional): Output directory.

--DEEP_FRAME (optional): Path of the deep frame.

--WAVL_MASK (optional): Wavelength masked for sky lines (default: [[7950,8006], [8020,8040], [8230,8280]]).

--sn_thre (optional): S/N threshold for source detection (default: 3).

