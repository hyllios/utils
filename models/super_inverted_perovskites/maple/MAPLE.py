# Notes:
# -  Assumes any required data normalization has already been done
# -  Can pass Y (desired response) instead of MR (model fit to Y) to make fitting MAPLE to datasets easy

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import numpy as np


class MAPLE:

    def __init__(self, X_train, MR_train, X_val, MR_val, fe_type="rf", n_estimators=200, max_features=0.5,
                 min_samples_leaf=10, regularization=0.001):

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
            fe = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
                                       max_features=max_features)
        elif fe_type == "gbrt":
            fe = GradientBoostingRegressor(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
                                           max_features=max_features, max_depth=None)
        else:
            print("Unknown FE type ", fe_type)
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
                splits = fe[i].tree_.feature  # -2 indicates leaf, index 0 is root
                if splits[0] != -2:
                    scores[splits[0]] += fe[i].tree_.impurity[0]  # impurity reduction not normalized per tree
        elif fe_type == "gbrt":
            for i in range(n_estimators):
                splits = fe[i, 0].tree_.feature  # -2 indicates leaf, index 0 is root
                if splits[0] != -2:
                    scores[splits[0]] += fe[i, 0].tree_.impurity[0]  # impurity reduction not normalized per tree
        self.feature_scores = scores
        mostImpFeats = np.argsort(-scores)

        # Find the number of features to use for MAPLE
        retain_best = 0
        rmse_best = np.inf
        for retain in range(1, num_features + 1):

            # Drop less important features for local regression
            X_train_p = np.delete(X_train, mostImpFeats[retain:], axis=1)
            X_val_p = np.delete(X_val, mostImpFeats[retain:], axis=1)

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
        self.X = np.delete(X_train, mostImpFeats[retain_best:], axis=1)

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
        x_p = np.delete(x, mostImpFeats[self.retain:], axis=1)

        curr_leaf_ids = self.fe.apply(x)[0]
        weights = self.training_point_weights(curr_leaf_ids)

        # Local linear model
        lr_model = Ridge(alpha=self.regularization)
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
        for i in range(
                n):  # The contents of this inner loop are similar to explain(): doesn't use the features selected by MAPLE or return as much information
            x = X[i, :].reshape(1, -1)

            curr_leaf_ids = self.fe.apply(x)[0]
            weights = self.training_point_weights(curr_leaf_ids)

            # Local linear model
            lr_model = Ridge(alpha=self.regularization)
            lr_model.fit(self.X_train, self.MR_train, weights)

            pred[i] = lr_model.predict(x)[0]

        return pred
