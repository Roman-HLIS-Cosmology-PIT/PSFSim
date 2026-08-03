"""Exercise all the functionality in the basis classes."""


import numpy as np
import psfsim.basis as bs


def test_figurebasis():
    """Test for the _FigureBasis base class."""

    _bs = bs._FigureBasis()
    x, y = np.meshgrid(np.linspace(-0.5, 0.5, 3), np.linspace(-0.5, 0.5, 3))
    dummy = _bs.basis(x, y)
    assert np.shape(dummy) == (3, 3, 1)
    assert np.all(np.abs(dummy - 1) < 1.0e-8)
    dummy = _bs.valid(x, y)
    assert np.shape(dummy) == (3, 3)
    assert np.all(dummy)


def test_validfcn():
    """Test validity range for the basis classes."""

    # Zernike basis
    z = bs.ZernikeBasis(10.0, 3)
    x, y = np.meshgrid(np.linspace(-8, 8, 3), np.linspace(-8, 8, 3))
    good = np.where(z.valid(x, y), 1, 0)
    assert np.all(good == np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))

    # Legendre basis
    lb = bs.LegendreBasis([5, 10, -1, 1], 2, 1)
    x, y = np.meshgrid(np.linspace(-8, 8, 3), np.linspace(-8, 8, 3))
    good = np.where(lb.valid(x, y), 1, 0)
    assert np.all(good == np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]]))

    # Legendre basis, max order
    lbmo = bs.LegendreBasisMaxOrder([5, 10, -1, 1], 2)
    x, y = np.meshgrid(np.linspace(-8, 8, 3), np.linspace(-8, 8, 3))
    good = np.where(lbmo.valid(x, y), 1, 0)
    assert np.all(good == np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]]))


def test_all_romanbasis():
    """Test Roman surface perturbations that aren't active in the current model."""

    p = {
        "M1": {"ORDER": 6, "SKIP": 2},
        "M2": {"ORDER": 6, "SKIP": 3},
        "FM1": {"ORDER": 8, "SKIP": 3},
        "FM2": {"ORDERX": 5, "ORDERY": 4},
        "M3": {"ORDERX": 5, "ORDERY": 4, "SKIP": 3},
        "FPA": {"ORDER": 1},
        "S1": {"ORDER": 6, "SKIP": 2},
    }
    basis_set_all = bs.RomanBasisSet(p)

    target = {
        "M1": 25,
        "M2": 22,
        "FM1": 75,
        "FM2": 30,
        "M3": 24,
        "WFI01": 3,
        "WFI02": 3,
        "WFI03": 3,
        "WFI04": 3,
        "WFI05": 3,
        "WFI06": 3,
        "WFI07": 3,
        "WFI08": 3,
        "WFI09": 3,
        "WFI10": 3,
        "WFI11": 3,
        "WFI12": 3,
        "WFI13": 3,
        "WFI14": 3,
        "WFI15": 3,
        "WFI16": 3,
        "WFI17": 3,
        "WFI18": 3,
        "S1": 25,
    }

    for k in target:
        assert target[k] == basis_set_all.basis[k].N

    assert basis_set_all.N == 255
