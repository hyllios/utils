import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from maple import MAPLE
from tqdm import trange
import pickle


def main(prop: str = None):
    N = 10
    errors = pd.DataFrame(columns=['training error', 'testing error'])
    models = []
    for i in trange(N):
        train_data = pd.read_csv(f'{prop}-training-sets/train{i}.dat', sep=' ')
        test_data = pd.read_csv(f'{prop}-training-sets/test{i}.dat', sep=' ')

        to_drop = ['comp']
        train_data.drop(to_drop, inplace=True, axis=1)
        test_data.drop(to_drop, inplace=True, axis=1)

        y_train = train_data[prop].to_numpy()
        X_train = train_data.drop(prop, axis=1).to_numpy()
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=.2)

        y_test = test_data[prop].to_numpy()
        X_test = test_data.drop(prop, axis=1).to_numpy()

        maple = MAPLE(X_train, y_train, X_val, y_val)
        models.append(maple)
        errors.loc[len(errors)] = [mean_absolute_error(y_train, maple.predict(X_train)),
                                   mean_absolute_error(y_test, maple.predict(X_test))]
    pickle.dump(models, open(f'MAPLE-results/maple-{prop}-models.pickle', 'wb'))
    return errors, models


if __name__ == '__main__':
    property = 'lambda'
    errors, models = main(property)
    errors.plot()
    plt.title(property)
    plt.ylabel('MAE')
    plt.xlabel('model number')
    print(errors)
    print(f"Mean Absolute Error = {errors['testing error'].mean()}")
    plt.savefig(f'MAPLE-results/{property}-maple-mae.pdf')

    features = ["structure.volume", "Acharge", "Bcharge", "Ccharge", "dos_ef", "AColumn", "BColumn", "CColumn", "ARow",
                "BRow", "CRow", "AElectronegativity", "BElectronegativity", "CElectronegativity", "AAtomicWeight",
                "BAtomicWeight", "CAtomicWeight", "ACovalentRadius", "BCovalentRadius", "CCovalentRadius",
                "ASqrtAtomicWeight", "BSqrtAtomicWeight", "CSqrtAtomicWeight"]
    scores = pd.DataFrame(columns=['features'])
    scores['features'] = features
    for i in range(len(models)):
        scores[f'scores{i + 1}'] = models[i].feature_scores
    scores['mean_score'] = scores.drop('features', axis=1).mean(axis=1)
    print(scores.sort_values(by='mean_score', ascending=False, ignore_index=True)[['features', 'mean_score']].head(6))

    plt.show()
