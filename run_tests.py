'''
This module was added to enable logging in unittest
module used by churn_script_logging_and_tests.py

Author: Andrey Baranov
'''
import logging
import unittest

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO,
                        filename="./logs/churn_script_logging_and_tests.log",
                        filemode='w',
                        format='%(name)s - %(levelname)s - %(message)s')

    loader = unittest.TestLoader()

    tests = loader.discover(pattern="*_tests.py",
                            start_dir=".")

    runner = unittest.runner.TextTestRunner()

    runner.run(tests)
