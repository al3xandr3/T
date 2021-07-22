    
__version__ = '0.1'

# Imported methods
from t.table import *

from t.stats import *

from t.eda import *

from t.financial import *

from t.ml import *

from t.db import *

# Shortcut for pipe operator
import pandas as pd
pd.DataFrame.p = pd.DataFrame.pipe
