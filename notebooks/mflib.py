import os
import time
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

plt.style.use('bmh')

COLORS = list(mcolors.CSS4_COLORS.keys())
SMALL_SIZE = 12
MEDIUM_SIZE = 14
LARGE_SIZE = 16

## plot style, fonts and colors
plt.style.use('seaborn')
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=LARGE_SIZE)   # fontsize of the figure title


def load_data(pth, names=None, n_data_sets=4):

    print("... loading data")
    if names is None:
        names = ['unit_nr', 'time', 'os_1', 'os_2', 'os_3']
        names += ['sensor_{0:02d}'.format(s + 1) for s in range(26)]

    dc = {}

    for i in range(n_data_sets):
        p = os.path.join(pth, 'RUL_FD00{}.txt'.format( i +1))
        df_RUL = pd.read_csv(p, sep= ' ', header=None, names=['RUL_actual'], index_col=False)
        p = os.path.join(pth, 'train_FD00{}.txt'.format( i +1))
        df_train = pd.read_csv(p, sep= ' ', header=None, names=names, index_col=False)
        p = os.path.join(pth, 'test_FD00{}.txt'.format( i +1))
        df_test = pd.read_csv(p, sep= ' ', header=None, names=names, index_col=False)
        s = 'FD_00{}'.format( i +1)
        dc[s] = {'df_RUL': df_RUL, 'df_train': df_train, 'df_test': df_test}


    return dc

def make_target(df, before_failure=10):
    """
    For each time stamp we can calculate RUL by subtracting the it from the total_runtime
    both RUL and runtime will be appended to the df
    :param df:
    :param before_failure:
    :return:
    """

    print("... making a needs maintenance target")
    total_runtime = np.zeros(df.shape[0])
    for unit in df['unit_nr'].unique():
        unit_inds = np.where(df['unit_nr'].values == unit)[0]
        total_runtime[unit_inds] = df.iloc[unit_inds, :]['time'].max()
    df['total_runtime'] = total_runtime
    df['RUL'] = df['total_runtime'] - df['time']
    df['needs_maintenance'] = [1 if r < before_failure else 0 for r in df['RUL']]
    print("...... orig data shape {} x {}".format(df.shape[0],df.shape[1]))

    return(df)

def munge_data(df, numeric_features, categorical_features):
    """

    Deal with missing values then use specific numeric and categorical features to create a feature matrix X, and
    a target vector y.

    :param df:
    :param numeric_features:
    :param categorical_features:
    :return:
    """

    print("... munging data")
    X = df.copy()
    orig_cols = list(X.columns).copy()

    # drop columns with too many nans
    X.dropna(axis='columns', inplace=True, thresh=int(round(0.5*X.shape[0])))
    dropped_by_nan = list(set(orig_cols).difference(set(X.columns)))
    numeric_features = list(set(numeric_features).difference(dropped_by_nan))
    print("...... # columns dropped due to excessive NaNs: {}".format(dropped_by_nan))

    # drop columns with very low variance
    dropped_by_variance = [f for f in numeric_features if X[f].values.var() < 0.0000001]
    X.drop(columns=dropped_by_variance, inplace=True)
    print("...... columns dropped based on variance: {}".format(dropped_by_variance))
    numeric_features = list(set(numeric_features).difference(set(dropped_by_variance)))

    # create the feature matrix
    X = X[numeric_features + categorical_features]
    y = df['needs_maintenance'].copy()
    print("...... # of columns explicitly not used: {}".format(df.shape[1] - X.shape[1]))
    print("...... feature matrix shape: {} x {}".format(X.shape[0], X.shape[1]))
    return(X,y,numeric_features)

def plot_subset(df, unit_subset, features):
    """
    Make a exploratory plot using specific units and specific features
    :param df:
    :param unit_subset:
    :param features:
    :return:
    """

    subset_mask = [True if df['unit_nr'].values[i] in unit_subset else False for i in range(df.shape[0])]
    df_subset = df[subset_mask]

    fig, axs = plt.subplots(len(features), len(unit_subset), figsize=(16, 8), sharex=True, sharey=False)
    for f, feature in enumerate(features):
        for u, unit in enumerate(df_subset['unit_nr'].unique()):
            ax = axs[f, u]
            ax.set_facecolor('black')
            df_unit = df_subset[df_subset['unit_nr'] == unit]
            ax.plot(df_unit['time'], df_unit[feature], color=COLORS[f])
            ax.xaxis.set_major_locator(plt.MaxNLocator(3))

            if f == 0:
                ax.set_title("unit-{}".format(unit))
            elif f == len(features)-1:
                ax.set_xlabel('Time')

            if u == 0:
                ax.set_ylabel(feature)
            else:
                ax.set_yticks([])

            ax.set_ylim((df_subset[feature].min(), df_subset[feature].max()))
            ax.set_xlim((df_subset['time'].min(), df_subset['time'].max()))

    plt.tight_layout()
    return(plt)

def get_preprocessor(numeric_features, categorical_features):
    """
    return a sklearn pipeline transformer
    :param categorical_features:
    :param numeric_features:
    :return:
    """

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

    return(preprocessor)


if __name__ == '__main__':

    # load data
    print('loading data...')
    data_dir = os.path.join("..", "data")
    all_data = load_data(data_dir)
    df = all_data['FD_001']['df_train'].copy()

    # make target
    df = make_target(df, before_failure=10)

    # munge data
    numeric_features = ['os_1', 'os_2', 'os_3'] + ['sensor_' + str(i).zfill(2) for i in range(2, 22)]
    categorical_features = ['unit_nr']
    X, y, numeric_features = munge_data(df, numeric_features, categorical_features)

    # EDA
    unit_subset = df['unit_nr'].unique()[:9]
    features = ['sensor_02', 'sensor_03', 'sensor_04', 'sensor_07', 'sensor_08']
    #plt = plot_subset(df, unit_subset, features)
    #plt.show()

    # model training
    print("... model training")
    preprocessor = get_preprocessor(numeric_features, categorical_features)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    target_names = ["no_maintenance", "yes_maintenance"]
    print("...... train", sorted(Counter(y_train).items()))
    print("...... test", sorted(Counter(y_test).items()))
    print("...... target names", target_names)

    time_start = time.time()
    pipe = Pipeline(steps=[('preprocessor', preprocessor),
                          ('sgd', SGDClassifier(class_weight='balanced'))])
    #
    param_grid = {'sgd__penalty': ['l2', 'l1', 'elasticnet'],
                  'sgd__loss': ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron']
    }

    grid = GridSearchCV(pipe, param_grid=param_grid, cv=5, n_jobs=-1)
    grid.fit(X_train, y_train)
    y_pred = grid.predict(X_test)

    ## model results
    print("... model results")
    print("...... train time", time.strftime('%H:%M:%S', time.gmtime(time.time() - time_start)))
    print("...... best parameters: ", grid.best_params_)
    print("...... model score: %.3f" % balanced_accuracy_score(y_test, y_pred))
    print('done')


