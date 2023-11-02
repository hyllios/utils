import sys
from modnet.preprocessing import MODData
import numpy as np
from pymatgen.core import Structure
import pandas as pd
import pickle


def predict_ensemble(models, moddata):
    preds = [i.predict(moddata) for i in models]
    return pd.DataFrame(np.mean(preds, axis=0), columns=['Tc', 'la', 'wlog'], index=moddata.structure_ids)


if __name__ == "__main__":
    test = sys.argv[1]
    df = pd.read_pickle(test)
    df["structure"] = df.structure.apply(Structure.from_dict)
    available =  ['dosef', 'debye']

    # featurize data
    md = MODData(materials = df['structure'])
    md.featurize(n_jobs=4)
    md = md.df_featurized.join(df.set_index(md.df_featurized.index)[[i for i in available if i in df.columns]])
    md = MODData(materials = [None for _ in range(len(md))],
                    df_featurized=md, structure_ids=df.mat_id.to_list()
                )
    # Save featurized dataset for future usage
    md.save('featurized_test_data')

    # Load the featurized data
    test = MODData.load('featurized_test_data')

    # Load the modnet models
    model = pickle.load(open("model.pk.bz2", "rb"))

    # print model CV errros
    cv_errors = np.mean([[a.history[i][-1] for i in ['val_tc_mae', 'val_la_mae', 'val_wlog_mae']] for a in model], axis=0)
    cv_errors_std = np.std([[a.history[i][-1] for i in ['val_tc_mae', 'val_la_mae', 'val_wlog_mae']] for a in model], axis=0)
    print(f"CV Tc error: {cv_errors[0]:.3f}±{cv_errors_std[0]:.3f}")    
    print(f"CV lambda error: {cv_errors[1]:.3f}±{cv_errors_std[1]:.3f}")    
    print(f"CV wlog error: {cv_errors[2]:.3f}±{cv_errors_std[2]:.3f}")    

    # get predictions
    df_test = predict_ensemble(model, test)
    df_test.index.name = "mat_id"
    df_test = df_test.reset_index()
    # save predictions to csv
    df_test.to_csv("predictions.csv", index=False)

    # calculate MAE
    df = df.merge(df_test, on="mat_id", suffixes=("","_pred"))
    for i in ["Tc", "la", "wlog"]:
        mae = (df[i] - df[f"{i}_pred"]).abs().mean()
        print(f"MAE in {i} (mean: {df[i].mean():.2f}): {mae:.2f}")



