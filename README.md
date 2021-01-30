![image](https://github.com/patternizer/pha-frontend/blob/master/PLOTS/ireland_stations_5000.png)

# pha-frontend

Python front-end for selection of land surface air temperature monitoring station subsets from CRUTEM5 for pairwise homogenisation algorithm (PHA) work for [GloSAT](https://www.glosat.org):

* python lasso code to filter by country, search radius and homogenisation lewvel
* python input file construction code to put extracted data subsets into the correct form for reading by NOAA's PHAv52[i,j] Fortran code

## Contents

* `pha-subsets.py` - python lasso code
* `pha-input.py` - python input file construction code

The first step is to clone the latest pha-frontend code and step into the check out directory: 

    $ git clone https://github.com/patternizer/pha-frontend.git
    $ cd pha-frontend

### Using Standard Python

The code should run with the [standard CPython](https://www.python.org/downloads/) installation and was tested in a conda virtual environment running a 64-bit version of Python 3.8+.

pha-frontend scripts can be run from sources directly, once the dependency packages are resolved.

Run with:

    $ python pha-subsets.py
    $ python pha-input.py

## License

The code is distributed under terms and conditions of the [Open Government License](http://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).

## Contact information

* [Michael Taylor](michael.a.taylor@uea.ac.uk)

