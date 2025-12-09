import math
import numpy as np  

from pcassie.utility_functions import *

def test_normalize_zero_one():
    test_arr = np.linspace(-1000, 1000, 10000)
    test_arr_normalized = normalize_zero_one(test_arr)
    tolerance = 1e-10
    assert math.fabs(np.nanmax(test_arr_normalized) - 1) < tolerance, \
        "pcassie.untility_functions.normalize_zero_one incorrext max value."
    assert math.fabs(np.nanmin(test_arr_normalized)) < tolerance, \
        "pcassie.untility_functions.normalize_zero_one incorrext min value."
    return None
    

    
