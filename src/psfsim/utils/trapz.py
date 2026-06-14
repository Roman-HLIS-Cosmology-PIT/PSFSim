"""General form of trapz."""

import numpy as np

_np_version = [int(x) for x in str(np.__version__).split(".")]
trapz = np.trapezoid if _np_version[0] >= 2 else np.trapz
