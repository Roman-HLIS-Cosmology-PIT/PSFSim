"""Basis functions for decomposition of the figure errors."""

import numpy as np
from scipy.special import legendre_p

from .zernike import noll_to_zernike, zernike


class _FigureBasis:
    """
    Base class for basis functions for figures.

    Parameters
    ----------
    None

    Attributes
    ----------
    N : int
        The number of basis modes.

    Methods
    -------
    basis
        Function to take an array: ``valid(x,y)``, where ``x`` and ``y`` are
        numpy arrays, and return an array with one extra axis (at the end) giving the
        basis mode index.
        Should be replaced when you inherit.
    valid
        Function to take an array: ``valid(x,y)``, where ``x`` and ``y`` are
        numpy arrays, and return a Boolean numpy array of whether it is valid.
        Should be replaced when you inherit.

    """

    def __init__(self):
        self.N = 1

    def basis(self, x, y):
        """
        Dummy basis function (all 1's).

        Parameters
        ----------
        x, y : np.ndarray of float
            The x and y coordinates of the points to evaluate.
            These should be the same shape.

        Returns
        -------
        np.ndarray of float
            Basis function array at these points; shape is ``np.shape(x) + (N,)``,
            where ``N`` is the number of basis functions.

        """

        return np.full(np.shape(x) + (1,), 1.0)

    def valid(self, x, y):
        """
        Dummy valid function (all True).

        Parameters
        ----------
        x, y : np.ndarray of float
            The x and y coordinates of the points to evaluate.
            These should be the same shape.

        Returns
        -------
        np.ndarray of bool
            True if valid, False if not; shape is the same as `x`.

        """

        return np.full(np.shape(x), True)


class ZernikeBasis(_FigureBasis):
    """
    Zernike basis set.

    Parameters
    ----------
    radius : float
        The maximum radius.
    nmax : int
        The maximum Zernike order.
    skip : int, optional
        If specified, skip orders below this value (e.g., skip=1 to not use piston).

    """

    def __init__(self, radius, nmax, skip=0):
        # inputs
        self.radius = radius
        self.nmax = nmax
        self.skip = skip

        # number of modes
        self.N = (nmax + 1) * (nmax + 2) // 2 - skip * (skip + 1) // 2

    def basis(self, x, y):
        """
        Zernike basis functions.

        These are normalized in a circle of radius ``self.radius``.

        Parameters
        ----------
        x, y : np.ndarray of float
            The x and y coordinates of the points to evaluate.
            These should be the same shape.

        Returns
        -------
        np.ndarray of float
            Basis function array at these points; shape is ``np.shape(x) + (N,)``,
            where ``N`` is the number of basis functions.

        """

        # polar coordinates
        rho = np.hypot(x, y) / self.radius
        theta = np.arctan2(y, x)

        # now build the basis
        out = np.zeros(np.shape(x) + (self.N,))
        offset = self.skip * (self.skip + 1) // 2
        for j in range(self.N):
            n, m = noll_to_zernike(j + 1 + offset)
            out[..., j] = zernike(n, m, rho, theta, normalized=True)
        return out

    def valid(self, x, y):
        """
        Valid function.

        This is True for points in a circle of radius ``self.radius``.

        Parameters
        ----------
        x, y : np.ndarray of float
            The x and y coordinates of the points to evaluate.
            These should be the same shape.

        Returns
        -------
        np.ndarray of bool
            True if valid, False if not; shape is the same as `x`.

        """

        return np.hypot(x, y) <= self.radius


class LegendreBasis(_FigureBasis):
    """
    Legendre basis set.

    Parameters
    ----------
    bbox : array-like of float
        The bounding box, in the form [xmin, xmax, ymin, ymax].
    nmax_x, nmax_y : int
        The maximum Legendre order on each axis.
    skip : int, optional
        Skip modes below this order.

    """

    def __init__(self, bbox, nmax_x, nmax_y, skip=0):
        # inputs
        self.xmin = bbox[0]
        self.xmax = bbox[1]
        self.ymin = bbox[2]
        self.ymax = bbox[3]
        self.nmax_x = nmax_x
        self.nmax_y = nmax_y
        self.skip = skip

        # number of modes
        self.N = (nmax_x + 1) * (nmax_y + 1) - skip * (skip + 1) // 2

    def basis(self, x, y):
        """
        The Legendre basis functions.

        These are organized by order in x (outer loop) and y (inner loop),
        and are normalized to rms=1.

        Parameters
        ----------
        x, y : np.ndarray of float
            The x and y coordinates of the points to evaluate.
            These should be the same shape.

        Returns
        -------
        np.ndarray of float
            Basis function array at these points; shape is ``np.shape(x) + (N,)``,
            where ``N`` is the number of basis functions.

        """

        out = np.zeros(np.shape(x) + (self.N,))
        u = 2.0 * (x - self.xmin) / (self.xmax - self.xmin) - 1.0
        v = 2.0 * (y - self.ymin) / (self.ymax - self.ymin) - 1.0
        ly = []
        for i in range(self.nmax_y + 1):
            ly.append(legendre_p(i, v) * np.sqrt(2 * i + 1))
        pos = 0
        for i in range(self.nmax_x + 1):
            lx = legendre_p(i, u) * np.sqrt(2 * i + 1)
            for j in range(self.nmax_y + 1):
                if i + j >= self.skip:
                    out[..., pos] = lx * ly[j]
                    pos += 1
        return out

    def valid(self, x, y):
        """
        Valid function.

        This is True for points in a circle of radius ``self.radius``.

        Parameters
        ----------
        x, y : np.ndarray of float
            The x and y coordinates of the points to evaluate.
            These should be the same shape.

        Returns
        -------
        np.ndarray of bool
            True if valid, False if not; shape is the same as `x`.

        """

        return x >= self.xmin & x <= self.xmax & y >= self.ymin & y <= self.ymax


class LegendreBasisMaxOrder(_FigureBasis):
    """
    Legendre basis set with maximum order.

    Parameters
    ----------
    bbox : array-like of float
        The bounding box, in the form [xmin, xmax, ymin, ymax].
    nmax : int
        The maximum Legendre order (total).
    skip : int, optional
        Skip modes below this order.

    """

    def __init__(self, bbox, nmax, skip=0):
        # inputs
        self.xmin = bbox[0]
        self.xmax = bbox[1]
        self.ymin = bbox[2]
        self.ymax = bbox[3]
        self.nmax = nmax
        self.skip = skip

        # number of modes
        self.N = (nmax + 1) * (nmax + 2) // 2 - skip * (skip + 1) // 2

    def basis(self, x, y):
        """
        The Legendre basis functions.

        These are organized by order in x (outer loop) and y (inner loop),
        and are normalized to rms=1.

        Parameters
        ----------
        x, y : np.ndarray of float
            The x and y coordinates of the points to evaluate.
            These should be the same shape.

        Returns
        -------
        np.ndarray of float
            Basis function array at these points; shape is ``np.shape(x) + (N,)``,
            where ``N`` is the number of basis functions.

        """

        out = np.zeros(np.shape(x) + (self.N,))
        u = 2.0 * (x - self.xmin) / (self.xmax - self.xmin) - 1.0
        v = 2.0 * (y - self.ymin) / (self.ymax - self.ymin) - 1.0
        ly = []
        for i in range(self.nmax + 1):
            ly.append(legendre_p(i, v) * np.sqrt(2 * i + 1))
        pos = 0
        for i in range(self.nmax + 1):
            lx = legendre_p(i, u) * np.sqrt(2 * i + 1)
            for j in range(self.nmax + 1):
                if i + j >= self.skip and i + j <= self.nmax:
                    out[..., pos] = lx * ly[j]
                    pos += 1
        return out

    def valid(self, x, y):
        """
        Valid function.

        This is True for points in a circle of radius ``self.radius``.

        Parameters
        ----------
        x, y : np.ndarray of float
            The x and y coordinates of the points to evaluate.
            These should be the same shape.

        Returns
        -------
        np.ndarray of bool
            True if valid, False if not; shape is the same as `x`.

        """

        return x >= self.xmin & x <= self.xmax & y >= self.ymin & y <= self.ymax


class RomanBasisSet:
    """
    Class to build a table of basis sets from a dictionary.

    Parameters
    ----------
    pars : dict
        The parameter dictionary. See below for parameters.

    Attributes
    ----------
    basis : _FigureBasis
        The basis functions (augmented with a ``start`` key for which index they start with).
    N : int
        Number of basis modes for the whole system.

    Notes
    -----
    The following parameters (nested if indicated) are allowed in `pars`:

    - ``M1`` : primary mirror
      - ``ORDER`` : Zernike order

    """

    def __init__(self, pars):
        self.basis = {}

        # Primary mirror
        if "M1" in pars:
            n1 = pars["M1"].get("ORDER", None)
            skip = pars["M1"].get("SKIP", 0)
            self.basis["M1"] = ZernikeBasis(1184.02, n1, skip=skip)

        # Secondary mirror
        if "M2" in pars:
            n1 = pars["M2"].get("ORDER", None)
            skip = pars["M2"].get("SKIP", 0)
            self.basis["M2"] = ZernikeBasis(266.255, n1, skip=skip)

        # 1st fold mirror
        if "FM1" in pars:
            n = pars["FM1"].get("ORDER", None)
            skip = pars["FM1"].get("SKIP", 0)
            self.basis["FM1"] = LegendreBasis([-151.11, 151.11, -123.58, 181.26], n, n, skip=skip)

        # 2nd fold mirror
        if "FM2" in pars:
            nx = pars["FM2"].get("ORDERX", None)
            ny = pars["FM2"].get("ORDERY", None)
            skip = pars["FM2"].get("SKIP", 0)
            self.basis["FM2"] = LegendreBasis([-216.955, 216.955, -139.97, 192.0], nx, ny, skip=skip)

        # Tertiary mirror
        if "M3" in pars:
            nx = pars["M3"].get("ORDERX", None)
            ny = pars["M3"].get("ORDERY", None)
            skip = pars["M3"].get("SKIP", 0)
            self.basis["M3"] = LegendreBasis([-302.715, 302.715, 15.285, 476.775], nx, ny, skip=skip)

        # FPA
        if "FPA" in pars:
            n = pars["FPA"].get("ORDER", None)
            skip = pars["FPA"].get("SKIP", 0)
            for chip in range(1, 19):
                self.basis[f"WFI{chip:02d}"] = LegendreBasisMaxOrder(
                    [-20.44, 20.44, -20.44, 20.44], n, skip=skip
                )

        # Filter -- put filter dependent parts last!
        if "S1" in pars:
            n1 = pars["S1"].get("ORDER", None)
            skip = pars["S1"].get("SKIP", 0)
            self.basis["S1"] = ZernikeBasis(52.65, n1, skip=skip)

        # Set up parameters
        self.__call__()

    def __call__(self):
        """Sets up the parameter mapping."""

        current = 0
        for j in self.basis:
            self.basis[j].start = current
            current += self.basis[j].N
        self.N = np.sum([self.basis[j].N for j in self.basis])


basis_set = RomanBasisSet(
    {
        "M1": {"ORDER": 6, "SKIP": 2},
        "M3": {"ORDERX": 5, "ORDERY": 4, "SKIP": 3},
        "FPA": {"ORDER": 1},
        "S1": {"ORDER": 6, "SKIP": 2},
    }
)

#        "M2": {"ORDER": 6, "SKIP": 3},
#        "FM1": {"ORDER": 8, "SKIP": 3},
#        "FM2": {"ORDERX": 5, "ORDERY": 4},
