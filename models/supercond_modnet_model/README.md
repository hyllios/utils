# Data

- *DS-A.pk.bz2*  and *DS-B.pk.bz2* are the training and test set used (pandas DataFrame);
- The columns are the id in the Alexandria database (mat_id), tc, la, wlog, structure, the density of states at the fermi level (dosef), and the predicted Debye temperature (debye);
- They can be openned with
```python
import pandas as pd

dsb = pd.read_pickle("DS-B.pk.bz2")
```
- *model.pk.bz2* contains the 5 models obtained during CV training;

# Predict

- In order to predict Tc, lambda and wlog for DS-B, run:
```bash
python predict.py DS-B.pk.bz2
```

- This will create a file (predictions.csv) with the predictions and calculate the mean absolute errors, using the mean of the 5 models described before; it also writes the CV errors during training.
