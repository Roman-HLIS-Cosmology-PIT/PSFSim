Surface error basis function treatment
######################################

The errors in the surface are described using functions in the ``psfsim.basis`` module. The basic organization is:

- The base class for basis functions is ``_FigureBasis``. The required attributes and methods are:

  - ``N`` (number of modes)

  - ``basis`` : Function to take arrays of ``x`` and ``y`` and return the basis functions with one extra axis at the end (to count which basis function).

  - ``valid`` : Function to take arrays of ``x`` and ``y`` and return a Boolean array of the same shape for whether they are in the domain.

- Derived classes are currently ``ZernikeBasis``, ``LegendreBasis``, and ``LegendreBasisMaxOrder``. The latter 2 differ by whether one uses Legendre polynomials up to some order (nx) in x and some order (ny) in y (``LegendreBasis``) or a maximum total order (``LegendreBasisMaxOrder``).

- The ``RomanBasisSet`` class contains a table of basis sets for each surface.

  - The ``basis`` attribute is a dictionary of ``_FigureBasis`` objects, with keys drawn from:

    - ``M1``, ``M2``, ``FM1``, ``FM2``, ``M3`` (for mirrors)
    - ``S1`` (for filter; may include ``S2`` in the future if we can ever tell the difference)
    - ``WFI01`` ... ``WFI18`` (detector surfaces)

  - The sense of these is that a *positive* value moves the surface away from the incoming light. Values are distances in mm for optics, and are distances divided by 8 * focalratio**2 for the detector surfaces.

- The ``basis_set`` object is an instance of the ``RomanBasisSet`` class and contains the default choice of basis for a simulation.
