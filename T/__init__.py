    
__version__ = '0.0.5'

# Imported methods
from t.table import *

from t.stats import *

from t.eda import *

from t.financial import *

from t.ml import *

# Shortcut for pipe operator
import pandas as pd
pd.DataFrame.p = pd.DataFrame.pipe