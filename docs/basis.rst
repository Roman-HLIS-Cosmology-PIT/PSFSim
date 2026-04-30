Surface error basis function treatment
######################################

``PSFSim`` also represents perturbations to the optical surfaces.

Organization
============

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

    Not every allowed surface needs to be in the basis.

  - The sense of these is that a *positive* value moves the surface away from the incoming light, and is measured in the direction normal to the local surface. Values are distances in mm for optics, and are distances divided by 8 * focalratio**2 for the detector surfaces.

- The ``basis_set`` object is an instance of the ``RomanBasisSet`` class and contains the default choice of basis for a simulation.

Basis choices
=============

All basis functions are functions of the (x, y) in the surface coordinates. Note many surface are rotated relative to the WFI or FPA plane!!

ZernikeBasis
------------

The Zernike basis modes are in a circle of radius R: Z_i(x/R, y/R) in Noll convention. So for example::

  Z_2(x, y) = 2 * x
  Z_3(x, y) = 2 * y
  Z_4(x, y) = np.sqrt(3) * (2 * (x**2 + y**2) / R**2 - 1)
  Z_5(x, y) = np.sqrt(6) * (2 * x * y) / R**2
  Z_6(x, y) = np.sqrt(6) * (x**2 - y**2) / R**2

You can specify an ``nmax`` (max order, inclusive) and a ``skip`` (skip below this order), so, e.g., if you have ``nmax=3`` and ``skip=2`` then the modes are

+-----------+------------+----------+----------------------------------+
| Index     | Noll index | (n, m)   | Name                             |
+-----------+------------+----------+----------------------------------+
|  0        |  4         | (2, 0)   | Focus                            |
+-----------+------------+----------+----------------------------------+
|  1        |  5         | (2,-2)   | Astigmatism +                    |
+-----------+------------+----------+----------------------------------+
|  2        |  6         | (2, 2)   | Astigmatism x                    |
+-----------+------------+----------+----------------------------------+
|  3        |  7         | (3,-1)   | Coma Y                           |
+-----------+------------+----------+----------------------------------+
|  4        |  8         | (3, 1)   | Coma X                           |
+-----------+------------+----------+----------------------------------+
|  5        |  9         | (3,-3)   | Trefoil Y                        |
+-----------+------------+----------+----------------------------------+
|  6        | 10         | (3, 3)   | Trefoil X                        |
+-----------+------------+----------+----------------------------------+

LegendreBasis and LegendreBasisMaxOrder
---------------------------------------

These bases is defined in a rectangle of some bounding box, xmin<=x<=xmax and ymin<=y<=ymax::

   u = 2 / (xmax-xmin) * ( x - (xmin+xmax)/2 )
   v = 2 / (xmax-xmin) * ( x - (xmin+xmax)/2 )
   basis_{l_x,l_y} = sqrt((2l_x+1)(2l_y+1)) P_{l_x}(u) P_{l_y}(v)

The ordering is that l_y is in the "inner" for loop and l_x is in the "outer" for loop.

The selection of modes is different for the two choices:

LegendreBasis
^^^^^^^^^^^^^

The choice of basis modes is given by ``nmax_x`` and ``nmax_y`` (inclusive) and a ``skip`` (skip below this order). So if you have ``nmax_x=3``, ``nmax_y=2``, and ``skip=1``, then the ordering of modes is

  (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2), (3,0), (3,1), (3,2)

LegendreBasisMaxOrder
^^^^^^^^^^^^^^^^^^^^^

The choice of basis modes is given by ``nmax`` (total order, inclusive) and a ``skip`` (skip below this order). So if you have ``nmax_x=3`` and ``skip=1``, then the ordering of modes is

  (0,1), (0,2), (0,3), (1,0), (1,1), (1,2), (2,0), (2,1), (3,0)


