"""Tests for Gaussian quadrature integration."""

import numpy as np
import pytest
from psfsim.psfobject import PSFObject
from psfsim.quadrature_integration import QuadratureIntegrator, build_exponential_decay_quadrature


class TestQuadratureIntegrator:
    """Test Gaussian quadrature integration."""

    def test_pure_exponential_integral(self):
        """
        Test quadrature on pure exponential: integral of e^(-alpha*z) from 0 to d.

        Analytical result: (1 - e^(-alpha*d)) / alpha
        """

        alpha = 1.1
        detector_thickness = 2.0  # microns
        n_order = 5
        z_nodes, z_weights = build_exponential_decay_quadrature(alpha, 0.0, detector_thickness, n_order)

        # Test with alpha = 1.0 (decay length = 1 micron)

        f_nodes = np.exp(-alpha * z_nodes)
        integral_quad = np.dot(f_nodes, z_weights)

        assert np.all(z_nodes >= 0), "Quadrature nodes should be non-negative"
        assert np.all(z_nodes <= detector_thickness + 1e-10), "Quadrature nodes exceed detector thickness"

        # All weights should be positive
        assert np.all(z_weights > 0), "Quadrature weights should be positive"

        # Test exponentially falling function
        x = np.sum(z_weights * np.exp(-2.0 * alpha * z_nodes))
        x_analytic = (1.0 - np.exp(-2.0 * alpha * detector_thickness)) / 2 / alpha
        assert np.abs(np.log(x / x_analytic)) < 1e-3

        # Analytical value
        integral_exact = (1.0 - np.exp(-alpha * detector_thickness)) / alpha

        # Check relative error < 1%
        rel_error = np.abs(integral_quad - integral_exact) / integral_exact
        assert rel_error < 1e-3, f"Quadrature error {rel_error:.2e} exceeds 0.1% for pure exponential"

    def test_integration_via_internal(self):
        """Tests the integration via QuadratureIntegrator."""

        u, v = np.meshgrid(np.linspace(0, 0.2, 8), np.linspace(0, 0.2, 8))
        q = QuadratureIntegrator(0.8, 5.0, u, v, None)

        # make an array of exponentials
        a = np.linspace(0.5, 5.0, 10)
        z_nodes = q.get_nodes_and_weights()[0]
        intensity = np.exp(-a[None, :] * z_nodes[:, None])
        sums = q.integrate(intensity, axis=0)
        integral_exact = (1.0 - np.exp(-a * 5.0)) / a
        err = np.abs(np.log(sums / integral_exact))
        print(err)
        assert err[0] < 0.1
        assert err[1] < 0.01
        assert err[2] < 0.002
        assert np.all(err[3:] < 2e-4)


class TestPSFObjectWithQuadrature:
    """Test PSFObject integration with new quadrature method."""

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

        assert obj.Optical_PSF.shape == (obj.ulen, obj.ulen)
        assert np.isclose(np.sum(obj.Optical_PSF), 1.0, rtol=1e-12, atol=1e-12)
        assert np.min(obj.Optical_PSF) >= -1e-10

        obj.get_Intensity_in_detector()

        assert hasattr(obj, "Intensity_in_detector"), "Should compute Intensity_in_detector"
        assert obj.Intensity_in_detector.shape == (obj.ulen, obj.ulen)
        assert np.all(np.isfinite(obj.Intensity_in_detector)), "Intensity should be finite"
        assert np.all(obj.Intensity_in_detector >= -1e-10), "Intensity should be non-negative"

        obj.get_image_from_Intensity()

        # Check results
        assert obj.detector_image.shape == (
            obj.postage_stamp_size * obj.ovsamp,
            obj.postage_stamp_size * obj.ovsamp,
        )
        assert np.all(obj.detector_image >= -1e-10), "Detector image should be non-negative"
        assert np.all(np.isfinite(obj.detector_image)), "Detector image should be finite"
        assert obj._quad_order < 20, "Quadrature order should be fewer than uniform trapezoid points"
        assert hasattr(obj, "_quadrature_integrator"), "PSFObject should have _quadrature_integrator"
        assert hasattr(obj, "_quad_nodes"), "PSFObject should have cached _quad_nodes"
        assert hasattr(obj, "_quad_weights"), "PSFObject should have cached _quad_weights"
        assert hasattr(obj, "_quad_order"), "PSFObject should have cached _quad_order"

        # Check shapes
        assert len(obj._quad_nodes) == len(obj._quad_weights), "Nodes and weights should have same length"
        assert obj._quad_order == len(obj._quad_nodes), "Order should match number of nodes"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
