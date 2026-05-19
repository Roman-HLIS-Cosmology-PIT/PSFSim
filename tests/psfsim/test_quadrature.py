"""Tests for Gaussian quadrature integration."""

import numpy as np
import pytest
from psfsim.filter_detector_properties import FilterDetector, n_mercadtel
from psfsim.index_cdte import n_cdte
from psfsim.psfobject import PSFObject
from psfsim.quadrature_integration import QuadratureIntegrator


class TestQuadratureIntegrator:
    """Test Gaussian quadrature integration."""

    def test_pure_exponential_integral(self):
        """
        Test quadrature on pure exponential: integral of e^(-alpha*z) from 0 to d.

        Analytical result: (1 - e^(-alpha*d)) / alpha
        """
        wavelength = 0.48  # microns
        detector_thickness = 2.0  # microns

        # Create minimal ux, uy for testing (just a grid)
        ux = np.zeros((2, 2))
        uy = np.zeros((2, 2))

        filter_obj = FilterDetector(
            [1.35, 1.82, 2.45, n_cdte(wavelength), n_mercadtel(wavelength)],
            [0.163, 0.137, 0.084, 0.010, 0.008],
            1,
        )

        integrator = QuadratureIntegrator(wavelength, detector_thickness, ux, uy, filter_obj)
        z_nodes, z_weights, order = integrator.get_nodes_and_weights()

        # Test with alpha = 1.0 (decay length = 1 micron)
        alpha = 1.0
        f_nodes = np.exp(-alpha * z_nodes)
        integral_quad = np.dot(f_nodes, z_weights)

        # Analytical value
        integral_exact = (1.0 - np.exp(-alpha * detector_thickness)) / alpha

        # Check relative error < 1%
        rel_error = np.abs(integral_quad - integral_exact) / integral_exact
        assert rel_error < 0.01, f"Quadrature error {rel_error:.2e} exceeds 1% for pure exponential"

        # Check that we're using a reasonable order
        assert 3 <= order <= 15, f"Quadrature order {order} out of expected range"

    def test_adaptive_order(self):
        """Test that adaptive order selection responds to decay length."""
        wavelength = 0.48
        detector_thickness = 2.0

        ux = np.zeros((4, 4))
        uy = np.zeros((4, 4))

        filter_obj = FilterDetector(
            [1.35, 1.82, 2.45, n_cdte(wavelength), n_mercadtel(wavelength)],
            [0.163, 0.137, 0.084, 0.010, 0.008],
            1,
        )

        integrator = QuadratureIntegrator(wavelength, detector_thickness, ux, uy, filter_obj)

        # Test _adaptive_order with different decay lengths
        order_sharp = integrator._adaptive_order(0.1)  # 0.1 / 2.0 = 0.05 << 0.15
        order_moderate = integrator._adaptive_order(0.5)  # 0.5 / 2.0 = 0.25 in [0.15, 0.3)
        order_slow = integrator._adaptive_order(3.0)  # 3.0 / 2.0 = 1.5 in [1.5, inf)

        # Sharp decay should use more points than slow decay
        assert order_sharp > order_moderate, "Sharp decay should use more quadrature points"
        assert order_moderate > order_slow, "Moderate decay should use more points than slow"

    def test_quadrature_nodes_in_range(self):
        """Test that quadrature nodes are within detector thickness."""
        wavelength = 0.5
        detector_thickness = 2.0

        ux = np.random.randn(8, 8) * 0.1  # Small random values
        uy = np.random.randn(8, 8) * 0.1

        filter_obj = FilterDetector(
            [1.35, 1.82, 2.45, n_cdte(wavelength), n_mercadtel(wavelength)],
            [0.163, 0.137, 0.084, 0.010, 0.008],
            1,
        )

        integrator = QuadratureIntegrator(wavelength, detector_thickness, ux, uy, filter_obj)
        z_nodes, z_weights, _ = integrator.get_nodes_and_weights()

        # All nodes should be within [0, detector_thickness]
        assert np.all(z_nodes >= 0), "Quadrature nodes should be non-negative"
        assert np.all(z_nodes <= detector_thickness + 1e-10), "Quadrature nodes exceed detector thickness"

        # All weights should be positive
        assert np.all(z_weights > 0), "Quadrature weights should be positive"


class TestPSFObjectWithQuadrature:
    """Test PSFObject integration with new quadrature method."""

    def test_psfobject_initialization(self):
        """Test that PSFObject initializes with quadrature integrator."""
        obj = PSFObject(
            4,
            20.15,
            5.12,
            wavelength=1.35,
            postage_stamp_size=31,
            ovsamp=8,
            detector_thickness=2.0,
            zlen=20,
            cycle=9,
        )

        # Check that quadrature integrator is initialized
        assert hasattr(obj, "_quadrature_integrator"), "PSFObject should have _quadrature_integrator"
        assert hasattr(obj, "_quad_nodes"), "PSFObject should have cached _quad_nodes"
        assert hasattr(obj, "_quad_weights"), "PSFObject should have cached _quad_weights"
        assert hasattr(obj, "_quad_order"), "PSFObject should have cached _quad_order"

        # Check shapes
        assert len(obj._quad_nodes) == len(obj._quad_weights), "Nodes and weights should have same length"
        assert obj._quad_order == len(obj._quad_nodes), "Order should match number of nodes"

    def test_psfobject_optical_psf(self):
        """Test that optical PSF computation works with quadrature integrator."""
        obj = PSFObject(
            4,
            20.15,
            5.12,
            wavelength=1.35,
            postage_stamp_size=31,
            ovsamp=8,
            cycle=9,
        )

        obj.get_optical_psf()
        assert obj.Optical_PSF.shape == (obj.ulen, obj.ulen)
        assert np.isclose(np.sum(obj.Optical_PSF), 1.0, rtol=1e-12, atol=1e-12)
        assert np.min(obj.Optical_PSF) >= -1e-10

    def test_psfobject_intensity_in_detector(self):
        """Test that intensity integration works with quadrature method."""
        obj = PSFObject(
            4,
            20.15,
            5.12,
            wavelength=1.35,
            postage_stamp_size=31,
            ovsamp=8,
            detector_thickness=2.0,
            cycle=9,
        )

        # This calls get_Intensity_from_E internally
        obj.get_Intensity_in_detector()

        assert hasattr(obj, "Intensity_in_detector"), "Should compute Intensity_in_detector"
        assert obj.Intensity_in_detector.shape == (obj.ulen, obj.ulen)
        assert np.all(np.isfinite(obj.Intensity_in_detector)), "Intensity should be finite"
        assert np.all(obj.Intensity_in_detector >= -1e-10), "Intensity should be non-negative"

    def test_psfobject_full_pipeline(self):
        """Test full PSF computation pipeline with quadrature integration."""
        obj = PSFObject(
            4,
            20.15,
            5.12,
            wavelength=1.35,
            postage_stamp_size=31,
            ovsamp=8,
            cycle=9,
        )

        # Run full pipeline
        obj.get_optical_psf()
        obj.get_Intensity_in_detector()
        obj.get_image_from_Intensity()

        # Check results
        assert obj.detector_image.shape == (
            obj.postage_stamp_size * obj.ovsamp,
            obj.postage_stamp_size * obj.ovsamp,
        )
        assert np.all(obj.detector_image >= -1e-10), "Detector image should be non-negative"
        assert np.all(np.isfinite(obj.detector_image)), "Detector image should be finite"
        assert obj._quad_order < 20, "Quadrature order should be fewer than uniform trapezoid points"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
