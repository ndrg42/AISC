#configuration setting for test
import os
os.getcwd()
import sys
sys.path.append('src/data')

#module needed for test or to test
import pytest
import pandas as pd

from DataLoader import SuperCon
#TO DO list for DataLoader:
#is a Dataframe?-> right type
#is in the correct format? -> chemical elements, formula,critical temperature

test_path = 'data/raw/supercon_tot.csv'

def test_supercon_type(test_path=test_path):
    """test supercon output type"""

    assert isinstance(SuperCon(test_path), type(pd.DataFrame()))
