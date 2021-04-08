_MuSeAnnotationBox_ a Python-based open-source toolkit for creating a variety of continuous and discrete emotion gold standards. In a single framework, we unify a wide range of fusion methods, such as Estimator Weighted Evaluator(EWE), DTW-Barycenter Averaging (DBA), and Generic-Canonical Time Warp-ing (GCTW), as well as providing an implementation of Rater Aligned Annotation Weighting (RAAW). The latter method, RAAW, aligns the annotations in an translation-invariant way before weighting them based on inter-rater agreement between the raw annotations. 

(c) Chair of Embedded Intelligence for Health Care and Wellbeing, University of Augsburg. Published under GNU General Public license, see LICENSE file for details. _TODO: add license_

Please direct any questions or requests to contact.muse2020[@]gmail.com or stappen[@]ieee.org or via PR.


# Citing
If you use MuSeFuseBox or any code from MuSeFuseBox in your research work, you are kindly asked to acknowledge the use 
of MuSeFuseBox in your publications. _TODO: citations_

> citation

```
@inproceedings{musefusebox,
...
}
```

# Installation


## Dependencies

* Python 3.7
* For CTW you need to install Octave 5.2.0 and check that 
    `os.environ['OCTAVE_EXECUTABLE'] == 'C:/Octave/Octave-5.2.0/mingw64/bin/octave-cli.exe'`

## Installing the python package
* We recommend the usage of a virtual environment for the MuSeFuseBox installation.
    ```bash 
    python3 -m venv muse_virtualenv
    ```
    Activate venv using:
    - Linux
    ```bash 
     source muse_virtualenv/bin/activate
    ```
    - Windows
    ```bash 
    muse_virtualenv\Scripts\activate.bat
    ```
    Deactivate with:
    ```bash 
     deactivate
    ```
* Once the virtual environment is activated, install the dependencies in requirements.txt with (later update with install using pip) _TODO: pip installation_
    ```bash 
    pip -r requirements.txt
    ```

The Installation is now complete.

# Configuration
Go through the tutorials for usage examples of the toolkit. _TODO: tutorials/examples_

Run with:
    ```
    python main.py [gold_standard, diarisation]
    ```
## Commandline Options: Gold Standard Generation
... _TODO: add cli options_
## Commandline Options: Diarisation
... _TODO: add cli options_
