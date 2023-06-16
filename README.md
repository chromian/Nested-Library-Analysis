# Nested-Library-Analysis
Nested-Library Analysis incorporating with Empirical Dynamic Modeling

The purpose of this project is to detect abrupt shifts in dynamics of chaotic natural systems with time series data.
Core algorithms of our novel method is provided in the folder CyNLA, in which Cython is required for the setup.

# Note
Since our method need to apply the chosen prediction method (here we use S-Map of EDM) over and over again while the training set is sequentially removed, whereas pyEDM runs slowly (probably because calling pandas everytime).
Thus, we used pyBindEDM, which is the wrapped module (from cppEDM in C/C++ to python) instead of pyEDM to reduce the computational cost.
This, meanwhile, makes our core algorithm may malfunction due to version issues about pyEDM provided by Sugihara's team.
Therefore, we upload the version of pyEDM we incorporate when building our module, and the files are in the folder pyEDM.
The purpose of it is troubleshooting.
There is no intention of plagiarization neither commercial usage of pyEDM.
