"""Script to describe field-dependent Zernikes in terms of surface basis functions.

Requires the directories:

- ``indata/`` : should have the Cycle 10 .csv files for WFI Imaging Mode.

- ``outdata/`` : where we write ``perturbations.csv``.

Note the output is in *nanometers* (multiply by 1e-6 to get mm), and this has to be re-run
if you change the basis set.

"""

import numpy as np
from psfsim.aberration_models import extract_basis_coefs
from psfsim.basis import basis_set

# remind us of where things start
for obj in basis_set.basis:
    b = basis_set.basis[obj]
    print(obj, b.start, b.N, b)

# filters
wfi_filters = {"H": "158", "R": "062", "Z": "087", "Y": "106", "J": "129", "F": "184", "K": "213", "W": "146"}
flts = list(wfi_filters.keys())
nfilters = len(flts)
nmodes = basis_set.N

data = np.zeros((nmodes, nfilters))

nmin = None
p = None
c = 3
for i in range(nfilters):
    filter = flts[i]

    print(f"\n---\nFILTER {filter}{wfi_filters[filter]}\n")
    p = extract_basis_coefs(
        f"indata/WIM_F{wfi_filters[filter]:s}_zernikes_cycle10.csv",
        filter,
        nn=64,
        pars_input=p,
        nmin=nmin,
        smin=0.05,
        c=c,
    )
    print(p)
    data[:, i] = p

    nmin = basis_set.basis["S1"].start
    c = 1

# save to a file, in *nanometers*
header = ",".join(flts)
np.savetxt("outdata/perturbations.csv", 1.0e6 * data, delimiter=",", fmt="%.4f", header=header)
