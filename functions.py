from pandas_profiling import ProfileReport
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats
from pandas_profiling import ProfileReport

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


# LOGS
def log(*m):
    print(" ".join(map(str, m)))


def black(s):
    return '\033[1;30m%s\033[m' % s


def green(s):
    return '\033[1;32m%s\033[m' % s


def red(s):
    return '\033[1;31m%s\033[m' % s


def yellow(s):
    return '\033[1;33m%s\033[m' % s


def to_dates(df, cols):
    """ Changes column format to datetime.

    Parameters:
    ----------
    df : dataframe
        Dataframe with columns which are falsely not recognised as datetime.

    cols : list
        list of columns, formats of which need to be corrected.

    Returns
    ----------
    df : dataframe with corrected column formats

    """
    for col in cols:
        df[col] = pd.to_datetime(df[col])
    return df


def get_randomdata(df, n=None, frac=None):
    """ Returns n or a fraction of randomly chosen rows.

    Parameters:
    ----------
    n : int, optional, default=None
        Number of items from axis to return. Cannot be used with `frac`.

    frac : float, optional, default=None
        Fraction of axis items to return. Cannot be used with `n`.

    Returns
    ----------
    df_sample : dataframe

    """
    if n is not None or frac is not None:
        # Randomly sample num_samples elements from dataframe
        df_sample = df.sample(n=n, frac=frac).iloc[:, 1:]
    else:
        df_sample = df.sample(n=100).iloc[:, 1:]
    return df_sample


def get_overview(df, n=None, max_rows=1000):
    """ Returns Pandas Profiling report.

    Parameters:
    ----------
    n : int, default=None
        Number of items from axis to return.

    max_rows : int, default=1000
        Number rows on which the ProfileReport is based.

    Notes
    ----------
    Due to technical limitations, the optimal maximum number of rows on which the report is based is 1000.
    If the actual number of rows is higher than 1000, then the report is constructed on randomly chosen 1000 rows.

    Returns
    ----------
    ProfileReport in html.

    """
    # max_rows = 1000  # the optimal maximum number of rows on which the report is based
    if n is None and df.shape[0] <= max_rows:
        return ProfileReport(df, title='Pandas Profiling Report', minimal=True, html={'style':{'full_width': True}})
    elif n is None and df.shape[0] > max_rows:
        print(f"Data is too large (> {max_rows} rows), getting overview for {max_rows} random samples")
        data = get_randomData(n=max_rows)
        return ProfileReport(data, title='Pandas Profiling Report', minimal=True, html={'style':{'full_width':True}})
    else:
        data = get_randomData(n=n)
        return ProfileReport(data, title='Pandas Profiling Report', minimal=True, html={'style':{'full_width':True}})


def get_summary(df,
                y,
                nan=False,
                formats=False,
                categorical=False,
                min_less_0=False,
                check_normdist=False,
                plot_boxplots=False):
    """Describes the data.

    Parameters:
    ----------
    df : DataFrame
        dataframe on which the summary will be based.

    y : Series
        response variable.

    nan : bool, default=True
        True if need to return a list of NaNs.

    formats : bool, default=True
        True if need to return all the formats of the columns.

    categorical : bool, default=True
        True if need to return values which can be categorical.
        Variable is considered to be categorical if there are less uique values than num_ifcategorical.

    min_less_0 : bool, default=True
        True if need check for variables which have negative values.

    check_normdist : bool, default=True
        True if need check actual distribution against Normal distribution.
        Will make plots of each variable considered against the Normal distribution.

    Returns
    ----------
    A description of the data in text format and plots if check_normdist=True.

    """
    #get numeric data only
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    df_numeric = df.select_dtypes(include=numerics)

    # Checking for NaN
    if nan:
        nans = list(
            pd.DataFrame(df.isna().sum()).rename(columns={0: 'NaNs'}).reset_index().query("NaNs>0")['index'])
        log(black('NaNs: '), nans)
    else:
        nans = False

    # Checking for unique formats
    if formats:
        unique_formats = list(df.dtypes.unique())
        log(black('Unique formats: '), unique_formats)
    else:
        formats is False

    # Checking for possible categorical values
    if categorical:
        num_ifcategorical = 10
        possibly_categorical = []
        for col in df.columns:
            set_unique = set(df[col])
            if len(set_unique) <= num_ifcategorical:
                possibly_categorical.append(col)
        log(black(f'Possible categorical variables (<{num_ifcategorical} unique values): '), possibly_categorical)
    else:
        categorical is False

    # Checking if min value is < 0
    if min_less_0:
        lst_less0 = list(
            pd.DataFrame(df_numeric[df_numeric < 0].any()).rename(columns={0: 'flag'}).query("flag==True").index)
        log(black(f'Min value < 0: '), lst_less0)
    else:
        min_less_0 is False

    # Plotting actual distributions vs Normal distribution
    def check_distribution(columns, plot_cols=6):
        plt.style.use('seaborn-white')

        if plot_cols > len(df_numeric.columns) - 2:
            # log(yellow('ERROR: '), f"Can't use more than {len(columns) - 2} columns.")
            plot_cols = len(df_numeric.columns) - 2
            if len(df_numeric.columns) - 2 < 3:
                plot_cols = len(df_numeric.columns)

        # figure size = (width,height)
        f1 = plt.figure(figsize=(30, len(df_numeric.columns) * 3))

        total_plots = len(df_numeric.columns)
        rows = total_plots - plot_cols +1

        for idx, y_var in enumerate(df_numeric.columns):
            if len(set(df_numeric[y_var])) >= 3:
                idx += 1
                ax1 = f1.add_subplot(rows, plot_cols, idx)
                ax1.set_xlabel(y_var)
                sns.distplot(df_numeric[y_var],
                             color='b',
                             hist=False
                             )
                # parameters for normal distribution
                x_min = df_numeric[y_var].min()
                x_max = df_numeric[y_var].max()
                mean = df_numeric[y_var].mean()
                std = df_numeric[y_var].std()
                # plotting normal distribution
                x = np.linspace(x_min, x_max, df_numeric.shape[0])
                y_var = scipy.stats.norm.pdf(x, mean, std)
                plt.plot(x, y_var, color='black', linestyle='dashed')

    if check_normdist:
        log(black('Plotting distributions of variables against normal distribution'))
        check_distribution(df_numeric.columns, plot_cols=6)


    # Plotting boxplots
    def boxplots(columns, plot_cols=6):
        """
        y - response variable column
        """
        plt.style.use('seaborn-white')

        col_types = ['datetime64[ns]']
        df_selected = df.select_dtypes(exclude=col_types)

        if plot_cols > len(df_selected.columns) - 2:
            # log(yellow('ERROR: '), f"Can't use more than {len(columns) - 2} columns.")
            plot_cols = len(df_selected.columns) - 2

        # figure size = (width,height)
        f1 = plt.figure(figsize=(30, len(df_selected.columns) * 3))

        total_plots = len(df_selected.columns)
        rows = total_plots - plot_cols

        df_x = df_selected.loc[:, ~df_selected.columns.isin([y.name])]

        for idx, x in enumerate(df_x):
            if len(set(df_selected[x])) >= 3 and (df_selected[x].dtype in numerics or y.dtype in numerics):
                idx += 1
                ax1 = f1.add_subplot(rows, plot_cols, idx)
                sns.boxplot(x=df[x], y=y, data=df_selected)

    if plot_boxplots:
        log(black('Plotting boxplots'))
        boxplots(df_numeric.columns, plot_cols=6)


#######################################################################################################################
# OUTLIERS
#######################################################################################################################
def z_score(df, columns, threshold=3):
    """Detects outliers based on z-score.

    Parameters:
    ----------
    columns : str
        A string of columns which will be analysed together using z-score.

    threshold : int, default=3
        Threshold against which the outliers are detected.

    Returns
    ----------
    df_outliers_clean : dataframe
        Dataframe without outliers.

    df_outliers : dataframe
        Dataframe of outliers.

    """
    # remove outliers based on chosen columns
    df_selected = df[columns].copy()

    # remove outliers
    z = np.abs(stats.zscore(df_selected))

    df_clean = df[(z < threshold).all(axis=1)]

    # get outliers df
    df_outliers = df[~df.index.isin(df_clean.index)]

    return df_clean, df_outliers

def IQR(df, columns, q1=0.25):
    """Detects outliers based on interquartile range (IQR).

    Parameters:
    ----------
    columns : str
        A string of columns which will be analysed together using IQR.

    q1 : float, default=0.25
        Threshold against which the outliers are detected.

    Returns
    ----------
    df_outliers_clean : dataframe
        Dataframe without outliers.

    df_outliers : dataframe
        Dataframe of outliers.

    """
    # remove outliers based on chosen columns
    print(columns)
    df_selected = df[columns]

    # remove outliers
    Q1 = df_selected.quantile(q1)
    Q3 = df_selected.quantile(1 - q1)
    IQR = Q3 - Q1

    df_clean = df[~((df_selected < (Q1 - 1.5 * IQR)) | (df_selected > (Q3 + 1.5 * IQR))).any(axis=1)]

    # get outliers df
    df_outliers = df[~df.index.isin(df_clean.index)]

    return df_clean, df_outliers


def plot(df, columns, df_clean, df_outliers, plot_cols=4):
    """Plots the dataframe and marks the outliers by a red cross.

    Parameters:
    ----------
    columns : str
        A string of columns which will be plotted.

    df_clean : dataframe
        Dataframe without outliers.

    df_outliers : dataframe
        Dataframe of outliers.

    plot_cols : int, default=6
        Determines how many columns the plots will form.

    """
    plt.style.use('seaborn-white')

    if plot_cols > len(columns) - 2:
        log(yellow('ERROR: '), f"Can't use more than {len(columns) - 2} columns in one row.")
        plot_cols = len(columns) - 2

    # figure size = (width,height)
    f1 = plt.figure(figsize=(30, len(columns) * 3))

    total_plots = len(columns)
    rows = total_plots - plot_cols

    for idx, y in enumerate(columns):
        idx += 1
        ax1 = f1.add_subplot(rows, plot_cols, idx)
        sns.regplot(x=df_clean.index,
                    y=y,
                    data=df_clean,
                    scatter=True,
                    fit_reg=False,
                    color='lightblue',
                    )
        sns.regplot(x=df_outliers.index,
                    y=y,
                    data=df_outliers,
                    scatter=True,
                    fit_reg=False,
                    marker='x',
                    color='red',
                    )


def show(df, columns, how='z_score', show_plot=False, **kwargs):
    """Detects outliers using one of the available methods.

    Parameters:
    ----------
    columns : str
        A string of columns which will be analysed together.

    how : str, default=z_score
        Method using which the outliers are detected.

    show_plot : bool, default=False
        True if need to see the plot of the data with the marked outliers.

    **kwargs
        Specifies extra arguments which may be necessary for one of the methods of finding outliers:

        threshold : int, default=3
            True if need to return all the formats of the columns.

        q1 : float, default=0.25
            True if need to return all the formats of the columns.

    Returns
    ----------
    df_clean : dataframe
        Dataframe without outliers.

    df_outliers : dataframe
        Dataframe of outliers.

    df : dataframe
        Original dataframe with outliers.
        Contains a new column called 'outliers' (bool) where the outliers are flagged (True if outlier).

    """
    if how == 'z_score':
        assert 'threshold' in kwargs, 'To use z-score method, threshold must be specified (default = 3)'
        df_clean, df_outliers = z_score(df, columns, kwargs['threshold'])
    elif how == 'IQR':
        assert 'q1' in kwargs, 'To use z-score method, q1 must be specified (default = 0.25)'
        df_clean, df_outliers = IQR(df, columns, kwargs['q1'])
    else:
        raise AttributeError('Unknown outlier detection method. Existing methods: z_score, IQR')

    df = df.copy()
    df['outliers'] = df.index.isin(df_outliers.index).copy()

    if show_plot:
        plot(df, columns, df_clean, df_outliers)

    return df_clean, df_outliers, df


#######################################################################################################################
# METRICS
#######################################################################################################################


def metrics(model, X_test, y_test):
    """
    Based on the count of each section, we can calculate precision and recall of each label:
    - __Precision__ is a measure of the accuracy provided that a class label has been predicted. It is defined by:
    precision = TP / (TP + FP)
    - __Recall__ is true positive rate. It is defined as: Recall =  TP / (TP + FN)
    So, we can calculate precision and recall of each class.
    __F1 score:__
    Now we are in the position to calculate the F1 scores for each label based on the precision and recall of that label
    The F1 score is the harmonic average of the precision and recall, where an F1 score reaches its best value at 1
    (perfect precision and recall) and worst at 0. It is a good way to show that a classifer has a good value for both
    recall and precision.
    """
    if type(model) is LogisticRegression:
        LR_yhat = model.predict(X_test)
        LR_yhat_prob = model.predict_proba(X_test)
        jss = jaccard_similarity_score(y_test, LR_yhat)
        f1s = f1_score(y_test, LR_yhat, average='weighted')
        lls = log_loss(y_test, LR_yhat_prob)
        print(str(type(model)).split('.')[-1][:-2])
        print("Jaccard index: %.2f" % jss)
        print("F1-score: %.2f" % f1s)
        print("LogLoss: %.2f" % lls)

    elif type(model) is KNeighborsClassifier or type(model) is DecisionTreeClassifier or type(model) is svm.SVC:
        yhat = model.predict(X_test)
        jss = jaccard_similarity_score(y_test, yhat)
        f1s = f1_score(y_test, yhat, average='weighted')
        print(str(type(model)).split('.')[-1][:-2])
        print("Jaccard index: %.2f" % jss)
        print("F1-score: %.2f" % f1s)

    print()

