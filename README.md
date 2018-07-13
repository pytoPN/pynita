# pynita

`pynita` is a Python package contains full collection of NITA implentation in Python. 

## Installation 

Since `pynita` has not been registered and distributed through `pypi.org`, user can install the package through cloned GitHub repository by following below steps: 

1. Clone the repository

    Clone `pynita` repository same way like you cloning any other project, you can use command line 

    ```
    git clone git@github.com:fengly20/pynita.git 
    ```

    Or use a Git client (e.g., GitHub Desktop, SourceTree, etc.).

    Note it's better to clone the repository and stay connected with GitHub since `pynita` will remain being actively developed rather than download the repository files as a zip file. 

2. Install the package using `pip`

    Before installing make sure you have `pip` in your python distribution. The `pip` should come with most Python distributions. `cd` into the repository directory then install it in develop mode:
   
    ```
    pip install -e .
    ```

    `pip` will automatically check and resolve the dependencies then install the package. 

## Use Example

### Python 

Once `pynita` has been successfully installed it could be imported into Python environment as same as importing other packages. 

```
from pynita import *
````

Then initialize the `nita` class. 

```
nita = nitaObj(user_ini)
```

### Command Line 

Full workflow must be developed using control file template in `./example`. Then the control file can be called from command line as: 

```
$python control_file.py -config_file [fullpathofconfigfile]
```

