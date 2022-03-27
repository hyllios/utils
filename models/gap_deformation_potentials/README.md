# Computational screening of materials with extreme gap deformation potentials
Code and Data used for machine learning in "Computational screening of materials with extreme gap deformation potentials".

For the CGCNN model use the data in data.tar.gz. It contains a folder full of cifs, and a list of the corresponding deformation potentials.

Install the CGCNN code from https://github.com/txie-93/cgcnn and run it with:
```bash
python main.py --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1 data/ --epochs 300 --batch-size 64 --learning-rate 0.01 --atom-fea-len 64 --h-fea-len 128  --optim Adam
```

For the MAPLE model use the data in maple_def_pot_data.tar.gz. The maple.py script trains a MAPLE model and calculates a validation error.

