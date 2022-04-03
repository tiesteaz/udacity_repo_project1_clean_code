'''
Test module for churn_library.py
Author: Andrey Baranov
Date: 31-Mar-2022
'''
import os
import glob
import unittest as ut
import logging as log
from datetime import date
import pandas as pd
import churn_library as cls


class TestImportData(ut.TestCase):
    """
    Test class for 'import_data' function
    Author: Andrey Baranov
    """

    def test_import_data_1_wrong_type(self):
        '''
        test data import - checks type of the input variable: pth
        OK: str
        FAIL: all other types
        '''
        # test wrong input paths, numerical input
        self.assertRaises(ValueError, cls.import_data, 123)
        self.assertRaises(ValueError, cls.import_data, 0)
        self.assertRaises(ValueError, cls.import_data, 12.3)
        self.assertRaises(ValueError, cls.import_data, -1)
        self.assertRaises(ValueError, cls.import_data, None)
        self.assertRaises(ValueError, cls.import_data, date.today())

    def test_import_data_2_wrong_path(self):
        '''
        test data import - checks that file exists at specified path
        OK: if file is found
        FAIL: if file is not found
        '''
        # test wrong input paths
        self.assertRaises(FileNotFoundError, cls.import_data, "123")
        self.assertRaises(FileNotFoundError, cls.import_data, "")
        self.assertRaises(FileNotFoundError, cls.import_data, '')

        log.info("test_import_data_2_wrong_path PASSED. %s\r\n", "")

    def test_import_data_3_file_with_invalid_data(self):
        '''
        test data import - checks that import fails if data in
        provided file is invalid
        '''
        # create temporary file with non-csv data
        with open('./data/bank_data_invalid.csv', 'w', encoding='utf-8') as text_file:
            text_file.write('some nonesense text')

        # try import and confirm that error is raised
        self.assertRaises(
            AssertionError,
            cls.import_data,
            "./data/bank_data_invalid.csv")

        # remove temporary file
        os.remove('./data/bank_data_invalid.csv')

        log.info("test_import_data_3_file_with_invalid_data PASSED. %s\r\n", "")

    def test_import_data_4_file_with_valid_data(self):
        '''
        test data import - checks type of the input variable: pth
        '''
        # test that input data file is converted into valid dataframe
        data_frame = cls.import_data("./data/bank_data.csv")

        # confirm that imported data is valid dataframe
        self.assertTrue(data_frame.shape[0] > 0 and data_frame.shape[1] > 0)

        log.info("test_import_data_4_file_with_valid_data PASSED. %s\r\n", "")


class TestEDA(ut.TestCase):
    '''
    Test class for exploratory data analysis (eda) function
    Author: Andrey Baranov
    '''
    log.basicConfig(
        filename='./logs/churn_script_logging_and_tests.log',
        level=log.INFO,
        filemode='w',
        format='%(name)s - %(levelname)s - %(message)s')

    @staticmethod
    def clear_working_folder():
        '''
        Removes all files recursively from the the working folder

        Returns: none
        '''
        filelist = glob.glob('./images/eda/*', recursive=True)
        for file in filelist:
            os.remove(file)

    def test_1_eda_text_output(self):
        '''
        Check that running perform_eda() creates at least one txt,
        or csv file in /images/eda folder.

        Steps:
        1. Create sample dataframe and pass it into perform_eda() function
        2. Check if at least one txt, csv, png file has
           been created in /images/eda folder

        Results:
         PASS - found at least one txt, csv, png file
         FAIL - haven't found any txt, csv, png files
        '''
        # clear all files in the output folder
        self.clear_working_folder()

        # Create test dataframe
        #test_data = {'Column1': [1,2], 'Column2': [3,4]}
        #test_data_frame = pd.DataFrame(data = test_data)
        test_data_frame = pd.read_csv(
            "./data/bank_data.csv",
            encoding="utf-8").head(10)

        # Run perform_eda() using test_data_frame
        cls.perform_eda(test_data_frame)

        # Check that at least one csv, txt or png file was created

        # list of extensions of the files expected from perform_eda()
        test_file_types = ['txt', 'png']

        for file_type in test_file_types:
            search_files = glob.glob(
                './images/eda/*.' + file_type,
                recursive=True)

            if len(search_files) > 0:
                for file in search_files:
                    error_msg = "perform_eda() produced empty file: {}".format(file)
                    # check that file of expected format is not empty
                    self.assertGreater(os.path.getsize(file), 0, error_msg)
            else:
                error_msg = "perform_eda() did not produce expected '{}' files".format(file_type)
                # check that at least one file of expected format was found
                self.assertGreater(len(search_files), 0, error_msg)

        log.info("test_1_eda_text_output PASSED. %s\r\n", "")


class TestEncoderHelper(ut.TestCase):
    '''
    Test class for 'encoder_helper' function
    Author: Andrey Baranov
    '''
    @staticmethod
    def clear_working_folder():
        '''
        Removes all files recursively from the the working folder

        Returns: none
        '''
        filelist = glob.glob('./images/eda/*', recursive=True)
        for file in filelist:
            os.remove(file)

    def test_encoder_helper_01(self):
        '''
        Check that encoder_helper() adds columns named as provided
        in response_category_lst
        '''
        self.clear_working_folder()
        # import small sample of the data (top 10 rows)
        test_data_frame = pd.read_csv(
            "./data/bank_data.csv",
            encoding="utf-8").head(22)

        # run encoder_helper function
        category_lst = ['Gender',
                        'Education_Level',
                        'Income_Category']
        response_category_lst = [
            'Gender_Chrn',
            'Education_Level_Chrn',
            'Income_Category_Chrn']

        # call encoder_helper() with 'response' parameter being passed into it
        encoded_dataframe = []
        encoded_dataframe = cls.encoder_helper(test_data_frame,
                                               category_lst,
                                               response=response_category_lst)

        difference = set(response_category_lst).difference(
            encoded_dataframe.columns)
        err_msg = "Categorized columns not found in \
                    the encoded dataframe, {}".format(difference)
        self.assertEqual(len(difference), 0, err_msg)

        log.info("test_encoder_helper_01 PASSED. %s\r\n", "")

    def test_encoder_helper_02(self):
        '''
        Check that default naming convention for
        encoded columns is '*_Churn'
        '''
        self.clear_working_folder()
        # import small sample of the data (top 10 rows)
        test_data_frame = pd.read_csv(
            "./data/bank_data.csv",
            encoding="utf-8").head(22)

        # run encoder_helper function
        category_lst = ['Gender',
                        'Education_Level',
                        'Income_Category']

        # call encoder_helper() without 'response' parameter being passed into
        # it
        encoded_dataframe = []
        encoded_dataframe = cls.encoder_helper(test_data_frame,
                                               category_lst)

        # create response_category_lst as default value: category + '_Churn'
        response_category_lst = [
            'Gender_Churn',
            'Education_Level_Churn',
            'Income_Category_Churn']

        difference = set(response_category_lst).difference(
            encoded_dataframe.columns)
        err_msg = "Categorized columns not found in the \
                    encoded dataframe, {}".format(
            difference)
        self.assertEqual(len(difference), 0, err_msg)

        log.info("test_encoder_helper_02 PASSED. %s\r\n", "")

    def test_encoder_helper_03(self):
        '''
        Check that encoder_helper() will throw error if category_lst
        contains columns
        that do not exist in the input dataframe
        '''
        self.clear_working_folder()
        # import small sample of the data (top 10 rows)
        test_data_frame = pd.read_csv(
            "./data/bank_data.csv",
            encoding="utf-8").head(22)

        # Set category_lst to a value that doesn't exist in the test data frame
        category_lst = ['Nationality']

        # Run encoder_helper and check that AssertionError is raised
        self.assertRaises(
            AssertionError,
            lambda: cls.encoder_helper(
                d_frame=test_data_frame,
                category_lst=category_lst))

        log.info("test_encoder_helper_03 PASSED. %s\r\n", "")

    def test_encoder_helper_04(self):
        '''
        Check that encoder_helper() will throw error if
        category_lst is not a list
        '''
        self.clear_working_folder()
        # import small sample of the data (top 10 rows)
        test_data_frame = pd.read_csv(
            "./data/bank_data.csv",
            encoding="utf-8").head(22)

        # Set category_lst to a value that is not a list
        category_lst = 'just a string'

        # Run encoder_helper and check that AssertionError is raised
        self.assertRaises(
            AssertionError,
            lambda: cls.encoder_helper(
                d_frame=test_data_frame,
                category_lst=category_lst))

        log.info("test_encoder_helper_04 PASSED. %s\r\n", "")

    def test_encoder_helper_05(self):
        '''
        Check that encoder_helper() will throw error if
        category_lst is not a list
        '''
        self.clear_working_folder()
        # import small sample of the data (top 10 rows)
        test_data_frame = pd.read_csv(
            "./data/bank_data.csv",
            encoding="utf-8").head(22)

        # Set category_lst to a value that is not a list
        category_lst = 2

        # Run encoder_helper and check that AssertionError is raised
        self.assertRaises(
            AssertionError,
            lambda: cls.encoder_helper(
                d_frame=test_data_frame,
                category_lst=category_lst))

        log.info("test_encoder_helper_05 PASSED. %s\r\n", "")

    def test_encoder_helper_06(self):
        '''
        Check that encoder_helper() will throw error if 'response'
        list size is not equal to the size of category lst
        '''
        self.clear_working_folder()
        # import small sample of the data (top 10 rows)
        test_data_frame = pd.read_csv(
            "./data/bank_data.csv",
            encoding="utf-8").head(22)

        # run encoder_helper function
        category_lst = ['Gender',
                        'Education_Level',
                        'Income_Category']
        response_category_lst = [
            'Gender_Chrn',
            'Education_Level_Chrn',
            'Income_Category_Chrn',
            'Marital_Status_Churn']

        # call encoder_helper() with 'response' parameter being passed into it
        # Run encoder_helper and check that AssertionError is raised
        self.assertRaises(
            AssertionError,
            lambda: cls.encoder_helper(
                d_frame=test_data_frame,
                category_lst=category_lst,
                response=response_category_lst))

        log.info("test_encoder_helper_06 PASSED. %s\r\n", "")

    def test_encoder_helper_07(self):
        '''
        Check that encoder_helper() will throw error if
        'response' is not a list, but string
        '''
        self.clear_working_folder()
        # import small sample of the data (top 10 rows)
        test_data_frame = pd.read_csv(
            "./data/bank_data.csv",
            encoding="utf-8").head(22)

        # run encoder_helper function
        category_lst = ['Gender', 'Education_Level', 'Income_Category']
        response_category_lst = 'Just a string'

        # call encoder_helper() with 'response' parameter being passed into it
        # Run encoder_helper and check that AssertionError is raised
        self.assertRaises(
            AssertionError,
            lambda: cls.encoder_helper(
                d_frame=test_data_frame,
                category_lst=category_lst,
                response=response_category_lst))

        log.info("test_encoder_helper_07 PASSED. %s\r\n", "")

    def test_encoder_helper_08(self):
        '''
        Check that encoder_helper() will throw error if 'response'
        is not a list, but int
        '''
        self.clear_working_folder()
        # import small sample of the data (top 10 rows)
        test_data_frame = pd.read_csv(
            "./data/bank_data.csv",
            encoding="utf-8").head(22)

        # run encoder_helper function
        category_lst = ['Gender', 'Education_Level', 'Income_Category']
        response_category_lst = 2

        # call encoder_helper() with 'response' parameter
        # Run encoder_helper and check that AssertionError is raised
        self.assertRaises(
            AssertionError,
            lambda: cls.encoder_helper(
                d_frame=test_data_frame,
                category_lst=category_lst,
                response=response_category_lst))

        log.info("test_encoder_helper_08 PASSED. %s\r\n", "")

    def test_encoder_helper_09(self):
        '''
        Check that encoder_helper() calculates mean Churn correctly
        '''
        self.clear_working_folder()
        # import small sample of the data (top 5 rows)
        test_data_frame = pd.read_csv(
            "./data/bank_data.csv",
            encoding="utf-8").head(5)

        category_lst = ['Gender']

        # prepare data:
        # Step 1: set all values in 'Gender' column to 'M'
        test_data_frame['Gender'].values[:] = 'M'

        # Step 2: set values in 'Gender' column to 'F' in rows 1 and 2
        test_data_frame.at[1, 'Gender'] = 'F'
        test_data_frame.at[2, 'Gender'] = 'F'

        # Step 3: set value of 'Attrition_Flag' not equal to
        # 'Existing Customer' in row 1
        # this will calculate 'Churn' = 1 for this row
        test_data_frame.at[1, 'Attrition_Flag'] = 'Lost Customer'

        # call encoder_helper()
        resulting_dataframe = cls.encoder_helper(d_frame=test_data_frame,
                                                 category_lst=category_lst)

        # Run encoder_helper and check calculated Gender_Churn = 1
        # Explanation: 2 of 5 rows has Gender = 'F' and
        # one of them has Churn = 1
        # i.e mean() Churn by Gender = 'F' will be 1/2 = 0.5
        self.assertEqual(resulting_dataframe.at[1, 'Gender_Churn'], 0.5)

        log.info("test_encoder_helper_09 PASSED. %s\r\n", "")

    def test_encoder_helper_10(self):
        '''
        Check that encoder_helper() calculates mean Churn correctly
        '''
        self.clear_working_folder()
        # import small sample of the data (top 5 rows)
        test_data_frame = pd.read_csv(
            "./data/bank_data.csv",
            encoding="utf-8").head(5)

        category_lst = ['Gender']

        # prepare data:
        # Step 1: set all values in 'Gender' column to 'F'
        test_data_frame['Gender'].values[:] = 'F'

        # Step 2: set value of 'Attrition_Flag' not equal
        # to 'Existing Customer' in row 1
        # this will calculate 'Churn' = 1 for this row
        test_data_frame.at[1:, 'Attrition_Flag'] = 'Lost Customer'

        # call encoder_helper()
        resulting_dataframe = cls.encoder_helper(d_frame=test_data_frame,
                                                 category_lst=category_lst)

        # Run encoder_helper and check calculated Gender_Churn = 0.8
        # Explanation: all 5 rows have Gender = 'F',
        # and 4 of 5 rows have churn = 1,
        # i.e mean() Churn by Gender = 'F' will be 4/5 = 0.8
        self.assertEqual(resulting_dataframe.at[1, 'Gender_Churn'], 0.8)

        log.info("test_encoder_helper_10 PASSED. %s\r\n", "")


class TestPerformFeatureEngineering(ut.TestCase):
    '''
    Test class for 'perform_feature_engineering' function
    Author: Andrey Baranov
    '''

    def test_perform_feature_engineering_1(self):
        '''
        Input parameter df must be pandas.DataFrame
        Pass wrong type of input parameter (dictionary) and check that AssertionError is returned.
        '''
        # Create wrong dataframe
        test_data = {'Column1': [1, 2], 'Column2': [3, 4]}

        self.assertRaises(
            AssertionError,
            lambda: cls.perform_feature_engineering(test_data))

        log.info("test_perform_feature_engineering_1 PASSED. %s\r\n", "")

    def test_perform_feature_engineering_2(self):
        '''
        Input parameter df must be pandas.DataFrame with valid columns
        Pass Dataframe missing expected columns and check that KeyError is raised.
        '''
        # Create wrong dataframe
        test_data = {'Column1': [1, 2], 'Column2': [3, 4]}

        # convert dictionary to pandas.DataFrame
        test_data_frame = pd.DataFrame(data=test_data)

        self.assertRaises(
            KeyError,
            lambda: cls.perform_feature_engineering(test_data_frame))

        log.info("test_perform_feature_engineering_2 PASSED. %s\r\n", "")

    def test_perform_feature_engineering_3(self):
        '''
        Input parameter df must be pandas.DataFrame with valid columns
        Pass valid Dataframe and verify that no error is returned.
        '''
        # valid data frame
        test_data_frame = pd.read_csv(
            "./data/bank_data.csv",
            encoding="utf-8").head(20)
        encoded_data = cls.encoder_helper(test_data_frame, ['Gender'])

        x_train, x_test, y_train, y_test = cls.perform_feature_engineering(
            encoded_data)

        # check that returned types are correct,
        # and that sum of rows in x_train and x_test = number of rows in encoded_data
        # and that sum of lengths of y_train and y_test = number of rows in
        # encoded_data
        self.assertTrue(
            isinstance(x_train, pd.DataFrame) and
            isinstance(x_test, pd.DataFrame) and
            isinstance(y_train, pd.Series) and
            isinstance(y_test, pd.Series) and
            x_train.shape[0] + x_test.shape[0] ==
            encoded_data.shape[0]
            and
            len(y_train) + len(y_test) ==
            encoded_data.shape[0])

        log.info("test_perform_feature_engineering_3 PASSED. %s\r\n", "")


class TestTrainModels(ut.TestCase):
    '''
    Test class for 'train_models' function
    Author: Andrey Baranov
    '''

    def test_train_models_1(self):
        '''
        pass less than required number of inputs and verify that error is returned
        '''
        self.assertRaises(AssertionError, lambda: cls.train_models('aaa',
                                                                   x_tst=None,
                                                                   y_trn=None,
                                                                   y_tst=None))

        log.info("test_train_models_1 PASSED. %s\r\n", "")

    def test_train_models_2(self):
        '''
        pass required number inputs of invalid types and verify that error is returned
        '''
        self.assertRaises(
            AssertionError, lambda: cls.train_models(
                'aaa', 'aaa', 'aaa', 1))

        log.info("test_train_models_2 PASSED. %s\r\n", "")

    def test_train_models_3(self):
        '''
        pass required number inputs of valid types,
        but with dataFrame with less than required number of classes of data
        '''
        # valid data frame
        test_data_frame = pd.read_csv(
            "./data/bank_data.csv",
            encoding="utf-8").head(20)
        encoded_data = cls.encoder_helper(test_data_frame, ['Gender'])

        # force all values of 'Churn' to 1, thus making only one class of data
        encoded_data['Churn'] = 1

        # print(encoded_data['Churn'])

        x_train, x_test, y_train, y_test = cls.perform_feature_engineering(
            encoded_data)

        self.assertRaises(
            ValueError, lambda: cls.train_models(
                x_train, x_test, y_train, y_test))

        log.info("test_train_models_3 PASSED. %s\r\n", "")

    def test_train_models_4(self):
        '''
        pass required number of valid inputs and verify that 4 files
        are saved in the /images/results folder
        '''
        # remove all *.png files in images/results/
        filelist = glob.glob(
            os.path.join(
                './images/results/',
                "*"),
            recursive=True)
        for file in filelist:
            os.remove(file)

        # valid data frame
        test_data_frame = pd.read_csv(
            "./data/bank_data.csv",
            encoding="utf-8").head(20)
        encoded_data = cls.encoder_helper(test_data_frame, ['Gender'])

        # force all values of 'Churn' to 1, thus making only one class of data
        encoded_data['Churn'] = 1
        encoded_data.at[10: encoded_data.columns.get_loc('Churn')] = 0

        # print(encoded_data['Churn'])

        x_train, x_test, y_train, y_test = cls.perform_feature_engineering(
            encoded_data)
        cls.train_models(x_train, x_test, y_train, y_test)

        file_1_exists = os.path.exists(
            './images/results/classification_report.png')
        file_2_exists = os.path.exists(
            './images/results/feature_importance_plot.png')
        file_3_exists = os.path.exists('./images/results/roc_curve_plot.png')
        file_4_exists = os.path.exists('./images/results/tree_explainer.png')

        self.assertTrue(file_1_exists and
                        file_2_exists and
                        file_3_exists and
                        file_4_exists
                        )

        log.info("test_train_models_4 PASSED. %s\r\n", "")


if __name__ == '__main__':

    ut.main()
