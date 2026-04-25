"""Basis functions for decomposition of the figure errors."""

import numpy as np

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

    """

    def __init__(self, radius, nmax):
        # inputs
        self.radius = radius
        self.nmax = nmax

        # number of modes
        self.N = (nmax + 1) * (nmax + 2) // 2

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
        for j in range(self.N):
            n, m = noll_to_zernike(j + 1)
            out[:, :, j] = zernike(n, m, rho, theta, normalized=True)
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


class RomanBasisSet:
    """
    Class to build a table of basis sets from a dictionary.

    The intended usage is for `pars` to come from a YAML file,
    though that isn't essential.

    Parameters
    ----------
    pars : dict
        The parameter dictionary. See below for parameters.

    Attributes
    ----------
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
            n1 = pars["M1"].get("ORDER", 5)
            self.basis["M1"] = ZernikeBasis(2370.0, n1)

        # Set up parameters
        self.__call__()

    def __call__(self):
        """Sets up the parameter mapping."""

        self.N = np.sum([self.basis[j].N for j in self.basis])
