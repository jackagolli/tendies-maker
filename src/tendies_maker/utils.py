import numpy as np


def pct_to_numeric(series):

    pct = series.values.astype('str')
    pct = np.char.strip(pct, chars='%')
    pct = pct.astype(np.float32)
    pct = pct / 100

    return pct
