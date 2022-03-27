from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import numpy as np
#from https://github.com/GDPlumb/MAPLE
class MAPLE:
 
    def __init__(self, X_train, MR_train, X_val, MR_val, fe_type = "rf", n_estimators = 200,random_state=0,
                 max_features = 0.5, min_samples_leaf = 10, regularization = 0.001):
        
        # Features and the target model response
        self.X_train = X_train
        self.MR_train = MR_train
        self.X_val = X_val
        self.MR_val = MR_val
        
        # Forest Ensemble Parameters
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        
        # Local Linear Model Parameters
        self.regularization = regularization
        
        # Data parameters
        num_features = X_train.shape[1]
        self.num_features = num_features
        num_train = X_train.shape[0]
        self.num_train = num_train
        num_val = X_val.shape[0]
        
        # Fit a Forest Ensemble to the model response
        if fe_type == "rf":
            fe = RandomForestRegressor(n_estimators = n_estimators, min_samples_leaf = min_samples_leaf,
                                       max_features = max_features, random_state=random_state)
        elif fe_type == "gbrt":
            fe = GradientBoostingRegressor(n_estimators = n_estimators, min_samples_leaf = min_samples_leaf,
                                           max_features = max_features, max_depth = None, random_state=random_state)
        else:
            print("Unknown FE type ", fe)
            import sys
            sys.exit(0)
        fe.fit(X_train, MR_train)
        self.fe = fe
        
        train_leaf_ids = fe.apply(X_train)
        self.train_leaf_ids = train_leaf_ids
        
        val_leaf_ids_list = fe.apply(X_val)
        
        # Compute the feature importances: Non-normalized @ Root
        scores = np.zeros(num_features)
        if fe_type == "rf":
            for i in range(n_estimators):
                splits = fe[i].tree_.feature #-2 indicates leaf, index 0 is root
                if splits[0] != -2:
                    scores[splits[0]] += fe[i].tree_.impurity[0] #impurity reduction not normalized per tree
        elif fe_type == "gbrt":
            for i in range(n_estimators):
                splits = fe[i, 0].tree_.feature #-2 indicates leaf, index 0 is root
                if splits[0] != -2:
                    scores[splits[0]] += fe[i, 0].tree_.impurity[0] #impurity reduction not normalized per tree
        self.feature_scores = scores
        mostImpFeats = np.argsort(-scores)
                
        # Find the number of features to use for MAPLE
        retain_best = 0
        rmse_best = np.inf
        for retain in range(1, num_features + 1):
            
            # Drop less important features for local regression
            X_train_p = np.delete(X_train, mostImpFeats[retain:], axis = 1)
            X_val_p = np.delete(X_val, mostImpFeats[retain:], axis = 1)
                        
            lr_predictions = np.empty([num_val], dtype=float)
            
            for i in range(num_val):
                
                weights = self.training_point_weights(val_leaf_ids_list[i])
                    
                # Local linear model
                lr_model = Ridge(alpha=regularization)
                lr_model.fit(X_train_p, MR_train, weights)
                lr_predictions[i] = lr_model.predict(X_val_p[i].reshape(1, -1))
            
            rmse_curr = np.sqrt(mean_squared_error(lr_predictions, MR_val))
            
            if rmse_curr < rmse_best:
                rmse_best = rmse_curr
                retain_best = retain
                
        self.retain = retain_best
        self.X = np.delete(X_train, mostImpFeats[retain_best:], axis = 1)
                
    def training_point_weights(self, instance_leaf_ids):
        weights = np.zeros(self.num_train)
        for i in range(self.n_estimators):
            # Get the PNNs for each tree (ones with the same leaf_id)
            PNNs_Leaf_Node = np.where(self.train_leaf_ids[:, i] == instance_leaf_ids[i])
            weights[PNNs_Leaf_Node] += 1.0 / len(PNNs_Leaf_Node[0])
        return weights
        
    def explain(self, x):
        
        x = x.reshape(1, -1)
        
        mostImpFeats = np.argsort(-self.feature_scores)
        x_p = np.delete(x, mostImpFeats[self.retain:], axis = 1)
        
        curr_leaf_ids = self.fe.apply(x)[0]
        weights = self.training_point_weights(curr_leaf_ids)
           
        # Local linear model
        lr_model = Ridge(alpha = self.regularization)
        lr_model.fit(self.X, self.MR_train, weights)

        # Get the model coeficients
        coefs = np.zeros(self.num_features + 1)
        coefs[0] = lr_model.intercept_
        coefs[np.sort(mostImpFeats[0:self.retain]) + 1] = lr_model.coef_
        
        # Get the prediction at this point
        prediction = lr_model.predict(x_p.reshape(1, -1))
        
        out = {}
        out["weights"] = weights
        out["coefs"] = coefs
        out["pred"] = prediction
        
        return out

    def predict(self, X):
        n = X.shape[0]
        pred = np.zeros(n)
        for i in range(n):
            exp = self.explain(X[i, :])
            pred[i] = exp["pred"][0]
        return pred

    # Make the predictions based on the forest ensemble (either random forest or gradient boosted regression tree) instead of MAPLE
    def predict_fe(self, X):
        return self.fe.predict(X)

    # Make the predictions based on SILO (no feature selection) instead of MAPLE
    def predict_silo(self, X):
        n = X.shape[0]
        pred = np.zeros(n)
        for i in range(n): #The contents of this inner loop are similar to explain(): doesn't use the features selected by MAPLE or return as much information
            x = X[i, :].reshape(1, -1)
        
            curr_leaf_ids = self.fe.apply(x)[0]
            weights = self.training_point_weights(curr_leaf_ids)
                    
            # Local linear model
            lr_model = Ridge(alpha = self.regularization)
            lr_model.fit(self.X_train, self.MR_train, weights)
                
            pred[i] = lr_model.predict(x)[0]
        
        return pred
        
def check_magnetic(struc):
    if any([element.is_lanthanoid and element.symbol!="La" for element in struc.composition]):
        return 0
    elif any([element.symbol in ['Mn','Cr','Fe','Ni'] for element in struc.composition]):
        return 0
    else:
        return 1
        

import re
import pandas as pd
import numpy as np
from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers import composition as cf
import pickle, gzip as gz
import sys,os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error as maef
import re

data = pickle.load(gz.open('maple_def_pot_data/def_pot2.pickle.gz','rb'))

feature_calculators = MultipleFeaturizer([cf.Stoichiometry(), cf.ElementProperty.from_preset("matminer"), cf.IonProperty(fast=True)])

feature_names = feature_calculators.feature_labels()

feature_names.append('not_magnetic')
feature_names.append('spacegroup')
feature_names.append('G')
feature_names.append('K')
feature_names.append('DFTorML')

features=[]
formulas_f = []
for el in data:
    f = feature_calculators.featurize(el['structure'].composition)
    f.extend([check_magnetic(el['structure'])])
    features.append(f)
    formulas_f.append(el["material_id"])
    
features2=[]
for el in data:
    try:
        features2.append([el['spacegroup']['number'], el['elasticity']['G_VRH'], el['elasticity']['G_VRH'], 1])
    except:
        features2.append([el['spacegroup']['number'], el['elastic'][0]['elastic_moduli']['G'], el['elastic'][0]['elastic_moduli']['K'], 0])


features = np.hstack([np.array(features),np.array(features2)])

features_c_efm = pickle.load(open('maple_def_pot_data/charges_and_effmass.pkl','rb'))
formulas_c = []
features_c=[]
for key, el in features_c_efm.items():
    if key not in [6475, 6475, 6475, 6475, 6475, 6475, 6475, 6475, 6475, 6475, 6476,
        6476, 6476, 6476, 6476, 6476, 6476, 6476, 6476, 6476, 6477, 6477,
        6477, 6477, 6477, 6477, 6477, 6477, 6477, 6477, 6478, 6478, 6478,
        6478, 6478, 6478, 6478, 6478, 6478, 6478, 6479, 6479, 6479, 6479,
        6479, 6479, 6479, 6479, 6479, 6479, 6480, 6480, 6480, 6480, 6480,
        6480, 6480, 6480, 6480, 6480, 6481, 6481, 6481, 6481, 6481, 6481,
        6481, 6481, 6481, 6481, 6482, 6482, 6482, 6482, 6482, 6482, 6482,
        6482, 6482, 6482]:
        try:
                name =  re.findall('mp.*',el['Key'])[0]
                formulas_c.append(name.replace('mp', 'mp-'))
        except:
            name =  re.findall('mvc.*',el['Key'])[0]
            formulas_c.append(name.replace('mvc', 'mvc-'))  
        try:
            features_c.append([np.min(el['bader']['1.0']), np.max(el['bader']['1.0']), np.std(el['bader']['1.0']),
                      np.mean(np.abs(el['bader']['1.0'])), np.min(el['ddec6']['1.0']), np.max(el['ddec6']['1.0']),
                      np.std(el['ddec6']['1.0']), np.mean(np.abs(el['ddec6']['1.0'])),el['eff_mass']['1.0']['h'],
                      el['eff_mass']['1.0']['e']])
        except:
            features_c.append([np.nan]*10)

features_c = np.array(features_c).astype(float)
formulas_c = np.array(formulas_c).reshape(-1,1)
formulas_c.shape, features_c.shape

df = pd.DataFrame(data=np.array(features), columns=feature_names)
df['formula'] = formulas_f
df = df[df['K'].notna()]
df = df[df['G'].notna()]
df = df.dropna(axis=1)


df_1 = pd.read_csv('maple_def_pot_data/gaps_der.csv')
names=[]
for i, el in enumerate(df_1['formula']):
    try:
        names.append(re.findall('mp.*',el)[0])
        df_1['formula'][i] = re.findall('mp.*',el)[0].replace('mp', 'mp-')
    except:
        names.append(re.findall('mvc.*',el)[0])
        df_1['formula'][i] = re.findall('mvc.*',el)[0].replace('mvc', 'mvc-')
    

names = [el.replace('mp', 'mp-') for el in names]
oxide_mp_ids = [el.replace('mvc', 'mvc-') for el in names]
dfa = pd.merge(df_1, df, on='formula')
dfa = pd.merge(dfa, pd.DataFrame(np.hstack((formulas_c, features_c)),
                                 columns=['formula', 'min_bader', 'max_bader', 'std_bader', 'ma_bader', 'min_ddec6',
                                         'max_ddec6', 'std_ddec6', 'ma_ddec6', 'eff_m_h', 'eff_m_e']), on='formula')

for key in dfa.keys():
    if key!='formula':
        dfa[key] = pd.to_numeric(dfa[key], downcast="float", errors='coerce')

dfa = dfa.dropna(1)
dfa = dfa.iloc[np.where(dfa['not_magnetic'])]
X_full =dfa.copy()
X_full.pop('gap_cont_vol')
X_full.pop('formula')
X_full.pop('cont_vol')
y_full = X_full.pop('def_pot')
X_full.pop('DFTorML')
X_full.pop('not_magnetic')
X_full['eff_m_e'] =  pd.to_numeric(X_full['eff_m_e'], downcast="float")
X_full.shape

mae_maple = []
feature_importances=[]
print("training")
for i in range(10):
    print(i)
    X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.2, random_state=i)
    X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=i+1)
    reg = MAPLE(X_train.to_numpy(), y_train.to_numpy(), X_valid.to_numpy(), y_valid.to_numpy(), fe_type = "rf", random_state=i)
    feature_importances.append((np.sort(reg.feature_scores)[np.where(np.sort(reg.feature_scores))], X_train.keys()[np.argsort(reg.feature_scores)][np.argwhere(np.sort(reg.feature_scores))]))
    mae_maple.append(maef(reg.predict(X_test.to_numpy()), y_test))

print(np.mean(mae_maple))
