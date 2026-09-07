* Espresso and wannier executables can be accessed by "module load quantum-espresso/7.3.1-gcc-13.2.0-6jwmo4k".

* The Python environment can be accessed by "source ~/software/venv/wannier/bin/activate". This is an uv environment, so use "uv install" to install new packages. Never use uv for testing, as it will copy its environment, taking hours. Just call pytest directly.

* All of core/ and analysis/ should be written in atomic units. Conversions should be done before entering and exiting these files. To simplify user's life, add a "to\_SI\_units(variable, physical\_quantity)" and a "to\_eVA\_units", etc. to units.py.

* workflows/w90tutorial should follow as closely as possible the wannier90 tutorials, including numbering.

* All band structures should plot E-EF, where EF is the Fermi energy
