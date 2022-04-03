# library doc string
'''
churn_library.py provides functions that can be used to import,
prepare data for model training,  train models, and save artifacts
of the training and trained models data in output folders.

Author: Andrey Baranov
Date: 31-Mar-2022
'''
# import libraries
import os
import glob
import logging as log
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import joblib
import shap
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from PIL import ImageDraw
from PIL import Image
mpl.use('Agg')


log.basicConfig(
    filename='./logs/churn_library.log',
    level=log.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            d_frame: pandas dataframe
    '''
    log.info("Importing data from: %s", pth)
    try:
        assert isinstance(pth, str)  # check that provided filepath is valid
    except AssertionError as exception:
        err = "Provided path value must be string\r\n"
        log.error(err)
        raise ValueError(err) from exception

    try:
        d_frame = pd.read_csv(pth, encoding='utf-8')
    except FileNotFoundError as exception:
        err = "File not found\r\n"
        log.error("%s\r\n", err)
        raise FileNotFoundError(err) from exception

    try:
        assert d_frame is not None
        assert d_frame.shape[0] > 0
        assert d_frame.shape[1] > 0
    except AssertionError as exception:
        err = "Imported data can't be converted into shape\r\n"
        log.error("%s\r\n", err)
        raise AssertionError(err) from exception


    log.info("Successfully imported data from %s \r\n", pth)
    return d_frame
#    pass


def perform_eda(d_frame):
    '''
    perform eda on df and save figures to /images/eda folder
    input:
            d_frame: pandas dataframe

    output:
            None
    '''

    work_folder = './images/eda'

    filelist = glob.glob(os.path.join(work_folder, "*"), recursive=True)
    for file in filelist:
        os.remove(file)

    print_output = ""

    print_output += "**** EDA #1 **** "
    eda_output = d_frame.shape
    print_output += "df.shape:\r\n{}\r\n\r\n".format(eda_output)

    print_output += "**** EDA #2 **** "
    eda_output = d_frame.isnull().sum()
    print_output += "df.isnull().sum():\r\n{} \r\n\r\n".format(eda_output)

    print_output += "**** EDA #3 **** "
    eda_output = d_frame.describe()
    print_output += "df.describe():\r\n{}\r\n\r\n".format(eda_output)

    print_output += "**** EDA #4 **** "
    eda_output = '\n'.join(map(str, [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]))
    print_output += "cat_columns:\r\n{}\r\n\r\n".format(eda_output)

    print_output += "**** EDA #5 **** "
    eda_output = '\n'.join(map(str, [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio'
    ]))
    print_output += "quant_columns:\r\n{}\r\n\r\n".format(eda_output)

    with open(work_folder + '/eda.txt', 'w', encoding='utf-8') as text_file:
        text_file.write(print_output)

    # define column 'Churn' = 1 if Column 'Attrition_Flaf' equals "Existing
    # Customer", otherwise = 0
    d_frame['Churn'] = d_frame['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    plt.clf()  # clears figure for new plot
    plt.rc('figure', figsize=(10, 10))
    d_frame['Churn'].hist()
    plt.savefig(fname=work_folder + '/churn_hist.png')

    plt.clf()
    plt.rc('figure', figsize=(10, 5))
    d_frame['Customer_Age'].hist()
    plt.savefig(fname=work_folder + '/customer_age_hist.png')

    plt.clf()
    plt.rc('figure', figsize=(10, 5))
    d_frame.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig(fname=work_folder + '/marital_status_hist.png')

    plt.clf()
    plt.rc('figure', figsize=(10, 5))
    # sns.distplot(df['Total_Trans_Ct'])
    sns.displot(d_frame['Total_Trans_Ct'])
    plt.savefig(fname=work_folder + '/Total_Trans_Ct_distribution.png')

    plt.clf()
    plt.rc('figure', figsize=(15, 15))
    sns.heatmap(d_frame.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig(fname=work_folder + '/correlation_heat_map.png')

    log.info("Successfully saved EDA data at %s \r\n", work_folder)


def encoder_helper(d_frame, category_lst, response=[]):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            d_frame: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional
                      argument that could be used for naming variables
                      or index y column]

    output:
            d_frame: pandas dataframe with new columns for
    '''

    # verify that input 'category_lst' is a list
    try:
        assert isinstance(category_lst, list)
    except AssertionError as exception:
        error_msg = "provided value of 'response' is not a list: {}".format(
            category_lst)
        log.error("%s\r\n", error_msg)
        raise AssertionError(error_msg) from exception

    # verify that input 'category_lst' is not empty
    try:
        assert len(category_lst) > 0
    except AssertionError as exception:
        error_msg = "input 'category_lst' is empty"
        raise AssertionError(error_msg) from exception

    # verify that all columns in category_lst exist in df
    try:
        assert len(set(category_lst).difference(d_frame.columns)) == 0
    except AssertionError as exception:
        missing_columns = set(category_lst).difference(d_frame.columns)
        error_msg = "Columns of category_lst not found in dataframe: {}".format(
            missing_columns)
        log.error("%s\r\n", error_msg)
        raise AssertionError(error_msg) from exception

    try:
        assert not isinstance(response, (float,int))
    except AssertionError as exception:
        err_msg = "wrong type of 'response' input variable, it must be list"
        log.error("%s\r\n", err_msg)
        raise AssertionError(err_msg) from exception

    if len(response) > 0:

        # verify that input 'response' is a list
        try:
            assert isinstance(response, list)
        except AssertionError as exception:
            error_msg = "provided value of 'response' is not a list: {}".format(
                response)
            log.error("%s\r\n", error_msg)
            raise AssertionError(error_msg) from exception

        # verify that size of 'response' is equal to the size of category_lst
        try:
            assert len(category_lst) == len(response)
        except AssertionError as exception:
            error_msg = "Size of provided 'response' list \
                         doesn't match size of the 'category_lst'\r\n"
            error_msg += "Size of the category_lst: {}\r\n".format(
                len(category_lst))
            error_msg += "Size of the response: {}\r\n".format(len(response))
            log.error("%s\r\n", error_msg)
            raise AssertionError(error_msg) from exception
    else:
        response = [s + '_Churn' for s in category_lst]

    d_frame['Churn'] = d_frame['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    for (category_column, category_response) in zip(category_lst, response):

        column_lst = []
        column_groups = d_frame.groupby(category_column).mean()['Churn']

        for val in d_frame[category_column]:
            column_lst.append(column_groups.loc[val])

        # add new columns to the dataframe
        d_frame[category_response] = column_lst

    df_cols = '\n'.join(map(str, d_frame.columns))
    with open('./images/eda/df_encoded_columns.txt', 'w', encoding='utf-8') as text_file:
        text_file.write(
            "Dataframe columns after encoding: \r\n\r\n{}".format(df_cols))

    d_frame.to_csv("./images/eda/df_encoded.csv")

    log.info("Data encoding has been completed successfully.\r\n")

    return d_frame

def perform_feature_engineering(d_frame, response=[]):
    '''
    input:
              d_frame: pandas dataframe
              response: string of response name [optional argument
                        that could be used for naming variables or
                        index y column]
    actions:
              Selects only numerical columns from the original dataframe,
              adds response columns from the provided 'response' list.

              Then performs train_test_split of the provided numberical data.

              Saves resulting scores at
              ./images/results/train_test_split_scores.txt
    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''

    try:
        assert isinstance(d_frame, pd.DataFrame)
    except AssertionError as exception:
        error_msg = "Provided input parameter 'd_frame' is not a pandas DataFrame\r\n"
        log.error("%s\r\n", error_msg)
        raise AssertionError(error_msg) from exception

    try:
        assert isinstance(response, list)
    except AssertionError as exception:
        error_msg = "Provided input parameter 'response' is not a list"
        log.error("%s\r\n", error_msg)
        raise AssertionError(error_msg) from exception

    if len(response) > 0:
        try:
            assert isinstance(response, list)
        except AssertionError as exception:
            error_msg = "Provided input parameter 'response' is not a list\r\n"
            log.error("%s\r\n", error_msg)
            raise AssertionError(error_msg) from exception

    X = pd.DataFrame()

    # Create list of columns of original dataframe containing only numerical
    # columns
    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio']

    # Add to this list response columns
    if len(response) > 0:
        keep_cols.extend(response)

    try:
        X[keep_cols] = d_frame[keep_cols]
    except KeyError as exception:
        err_msg = "invalid keys"
        log.error("%s\r\n", err_msg)
        raise KeyError(err_msg) from exception

    y = d_frame['Churn']

    # train test split
    x_trn, x_tst, y_trn, y_tst = train_test_split(
        X, y, test_size=0.3, random_state=42)

    log.info("Feature engineering has been completed successfully.\r\n")
    return x_trn, x_tst, y_trn, y_tst


def classification_report_image(y_trn,
                                y_tst,
                                y_trn_preds_lr,
                                y_trn_preds_rf,
                                y_tst_preds_lr,
                                y_tst_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_trin: training response values
            y_tst:  test response values
            y_trn_preds_lr: training predictions from logistic regression
            y_trn_preds_rf: training predictions from random forest
            y_tst_preds_lr: test predictions from logistic regression
            y_tst_preds_rf: test predictions from random forest

    output:
             None
    '''

    img = Image.new('RGB', (1000, 1000))
    lr_img = ImageDraw.Draw(img)
    lr_img.text((10, 10), "Classification Report")

    # render text output of the report
    cr_text = "{}\r\n\r\n".format('Classification Report:')

    cr_text += "{}\r\n".format('random forest results')
    cr_text += "{}\r\n".format('test results')
    cr_text += "{}\r\n\r\n".format(
        classification_report(
            y_tst, y_tst_preds_rf))

    cr_text += "{}\r\n".format('train results')
    cr_text += "{}\r\n\r\n".format(
        classification_report(
            y_trn, y_trn_preds_rf))

    cr_text += "{}\r\n\r\n".format('logistic regression results')
    cr_text += "{}\r\n".format('test results')
    cr_text += "{}\r\n\r\n".format(
        classification_report(
            y_tst, y_tst_preds_lr))

    cr_text += "{}\r\n".format('train results')
    cr_text += "{}\r\n".format(classification_report(y_trn,
                               y_trn_preds_lr))

    lr_img.text((10, 10), cr_text)

    work_folder = './images/results'
    img.save(work_folder + '/classification_report.png')

    log.info("Saved classification report at ./images/results/classification_report.png.\r\n")


def feature_importance_plot(model, x_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            x_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    cv_rfc = model

    X = pd.DataFrame()
    X = x_data

    # Calculate feature importances
    importances = cv_rfc.best_estimator_.feature_importances_

    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X.shape[1]), names, rotation=90)

    plt.savefig(fname=output_pth)

    log.info("Saved feature importance report at %s\r\n", output_pth)


def train_models(x_trn, x_tst, y_trn, y_tst):
    '''
    train, store model results: images + scores, and store models
    input:
              x_trn: X training data
              x_tst: X testing data
              y_trn: y training data
              y_tst: y testing data
    output:
              None
    '''
    try:
        assert isinstance(x_trn, pd.DataFrame)
    except AssertionError as exception:
        error_msg = "Provided input parameter 'x_trn' is not a pandas DataFrame\r\n"
        log.error("%s\r\n", error_msg)
        raise AssertionError(error_msg) from exception

    try:
        assert isinstance(x_tst, pd.DataFrame)
    except AssertionError as exception:
        error_msg = "Provided input parameter 'x_tst' is not a pandas DataFrame\r\n"
        log.error("%s\r\n", error_msg)
        raise AssertionError(error_msg) from exception

    try:
        assert isinstance(y_trn, pd.Series)
    except AssertionError as exception:
        error_msg = "Provided input parameter 'y_trn' is not a Series\r\n"
        log.error("%s\r\n", error_msg)
        raise AssertionError(error_msg) from exception

    try:
        assert isinstance(y_tst, pd.Series)
    except AssertionError as exception:
        error_msg = "Provided input parameter 'y_tst' is not a Series\r\n"
        log.error("%s\r\n", error_msg)
        raise AssertionError(error_msg) from exception

    # grid search
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression(solver='liblinear')

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(x_trn, y_trn)

    lrc.fit(x_trn, y_trn)

    y_trn_preds_rf = cv_rfc.best_estimator_.predict(x_trn)
    y_tst_preds_rf = cv_rfc.best_estimator_.predict(x_tst)

    y_trn_preds_lr = lrc.predict(x_trn)
    y_tst_preds_lr = lrc.predict(x_tst)

    # save scores
    classification_report_image(y_trn,
                                y_tst,
                                y_trn_preds_lr,
                                y_trn_preds_rf,
                                y_tst_preds_lr,
                                y_tst_preds_rf)

    # save feature importance plot
    feature_importance_plot(
        model=cv_rfc,
        x_data=x_tst,
        output_pth='./images/results/feature_importance_plot.png')

    # save best model
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')

    # save ROC Curve plot
    lrc_plot = plot_roc_curve(lrc, x_tst, y_tst)

    # plots
    # roc curve comparison (linear regression and random forest)
    plt.clf()
    plt.figure(figsize=(15, 8))
    v_ax = plt.gca()

    plot_roc_curve(
        cv_rfc.best_estimator_,
        x_tst,
        y_tst,
        ax=v_ax,
        alpha=0.8)
    lrc_plot.plot(ax=v_ax, alpha=0.8)
    plt.savefig(fname='./images/results/roc_curve_plot.png')

    # tree explainer
    plt.clf()
    explainer = shap.TreeExplainer(cv_rfc.best_estimator_)
    shap_values = explainer.shap_values(x_tst)
    shap.summary_plot(shap_values, x_tst, plot_type="bar")
    plt.savefig(fname='./images/results/tree_explainer.png')

    log.info("Model training complete, see results at ./images/results/ \r\n")

if __name__ == "__main__":

    # Step 1: import data
    data_frame = import_data("./data/bank_data.csv")

    # Step 2: perform exploratory data analysis
    perform_eda(data_frame)

    # Step 3: add columns with Churn by provided categories
    df_with_churn_columns = encoder_helper(data_frame,
                                           ['Gender',
                                            'Education_Level',
                                            'Marital_Status',
                                            'Income_Category',
                                            'Card_Category'],
                                           ['Gender_Churn',
                                            'Education_Level_Churn',
                                            'Marital_Status_Churn',
                                            'Income_Category_Churn',
                                            'Card_Category_Churn'])

    # Step 4: perform  feature engineering
    X_train, X_test, y_train, y_test = perform_feature_engineering(df_with_churn_columns.head(100),
                                                                   ['Gender_Churn',
                                                                    'Education_Level_Churn',
                                                                    'Marital_Status_Churn',
                                                                    'Income_Category_Churn',
                                                                    'Card_Category_Churn'])

    # Step 5: train models and save results
    train_models(X_train, X_test, y_train, y_test)
