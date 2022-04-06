# Machine-learning correction to density-functional crystal structure optimization 

This code was used  for "Machine-learning correction to density-functional crystal structure optimization" [https://doi.org/10.48550/arXiv.2111.02206].

*Proposed steps for running the code*

ensure required python packages
* json, joblib, bz2, fastparquet
* numpy, pandas
* matminer, pymatgen, scikit-learn 

provide external dependences
* create the subdirectory *external* and copy into it the MAPLE implementation *MAPLE.py* of Plumb et al. [https://github.com/GDPlumb/MAPLE, arXiv:1807.02910]
* copy the compressed json file *2021.04.06_ps.json.bz2* [https://doi.org/10.24435/materialscloud:ka-br ] with the PBEsol calculations of Schmidt et al. into the directory *external*
* run `python3 provideDataForm.py` to generate the file *external/ICSDParams.csv* containing the required ICSD-IDs
* complement this form with the missing experimental volumes (volume_ICSD) and temperatures (TInK_ICSD)

compile features
* run `python3 compileInputFeatures.py <MaterialsProjectAPIKey>` to generate the training features *external/matFeature.parquet* with *\<MaterialsProjectAPIKey>* being the user's  API Key for accessing data of the Materials Project via MPRester

train models & predict volumes
* run `python3 predictVolumes.py` to train & evaluate the models. The predicted volumes (\*.dat) are exported together with the feature importances (\*.ft) and trained models (\*.md) into the directory *data*
