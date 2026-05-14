The model of the optical system
###############################

``PSFSim`` models the PSF of the Roman WFI imaging mode. The basic philosophy of the code is to be "as 
close to first principles as possible." A few points on what this means are in order:

- First-principles tools like ``PSFSim`` are useful for exploring the importance of non-ideal physical 
  effects, or testing what needs to be in simplified models. They are a complement to faster 
  approximations to diffractive PSFs (e.g., wavefront expanded in Zernike modes with a single image plane 
  FFT), or purely empirical models. All three will be needed to achieve the calibration needs for the 
  Roman HLWAS imaging survey in reasonable time.

- As with all optical systems, the as-built Roman telescope differs at some level from the design 
  configuration (and the "true" configuration will even be time-dependent). We treat this with 
  perturbations to the optical system. Note that many combinations of motions or distortions of surfaces 
  are degenerate --- these directions in perturbation space that lead to no effect on the PSF or geometric 
  distortion can neither be constrained nor do they matter for the science, and so the perturbation 
  treatment breaks this degeneracy by artificially removing components. A consequence is that the 
  perturbations derived from ``PSFSim`` **cannot** be used to infer the physical displacement of any 
  components in the Roman payload.

- As an analysis tool, ``PSFSim`` only uses information that can be shared and published as needed to 
  support the scientific results of the Roman mission. However, some information that in principle might 
  go into a physical model is proprietary to the contractors. In these cases, ``PSFSim`` uses approximate 
  models that --- like the perturbation parameters above --- will be updated to reproduce the observed 
  behavior of the system. (An example would be the use of a model coating with parameters tuned 
  empirically to produce the correct wavelength dependence, which serves the science analysis need even 
  though the model does not contain the true choice of materials.)

The "base" optical model
########################

The base model in ``psfsim.romantrace`` is based on the Roman Opto-Mechanical Definition. The 
``RomanRayBundle`` object consists of a bundle of rays (with positions, directions of propagation, a 
mask, and electric field vectors) that are initialized in an entrance plane and then traced through the 
sequence of reflections until they reach the Focal Plane. The electric field can be attenuated at each 
reflection by wavelength-dependent S- and P-polarized reflection amplitudes. (The treatment of the 
filter surfaces is *under construction*.)

At the focal plane, each ray is replaced with a plane wave, described by an incoming amplitude and 
direction cosines (u, v) with respect to the focal plane. This is propagated into a 3-dimensional 
electric field in the absorbing medium (a function of u, v, and z, where u/lambda_vac and v/lambda_vac 
are now the wave vectors in the detector plane, and z is the depth in the detector). This is then 
Fourier transformed to (x, y, z) and squared to give the power dissipation in the detector. Charge 
diffusion (right now using a sech function) is then applied.

Perturbations
#############

Perturbations to the optical system are described in two ways:

- Some perturbations (in ``psfsim.offsets``) describe solid-body translations and rotations.

- Other perturbations (in ``psfsim.basis``) represent surface errors expanded in basis functions on the 
  surface (usually Zernike or Legendre polynomials, depending on whether the natural domain is circular or 
  rectangular). The choices of basis function are described on the `basis page <basis.rst>`_.

The scripts that computed these are in the folders (note these scripts are not run during normal operation of PSFSim, they are just used to generate distortion files):

- `Cycle 10 <../scripts/cycle10/>`_

  - `fit_offsets.py <../scripts/cycle10/fit_offsets.py>`_ : Fits the 9 offset parameters (right now: 2 field bias, 1 SM position, 3+3 FPA translation + rotation) based on the astrometric distortion map and focus.

  - `convert_cycle10.py <../scripts/cycle10/convert_cycle10.py>`_ : Generates the best-fit surface perturbations. (The resulting files are in ``src/psfsim/data/cycle10_perturbations.csv``; that is a csv file where each column represents a filter band, and the rows represent each of the surface perturbation modes; there are 130 in total for Cycle 10.)

Surface coatings
################

The "true" coating model for the mirrors, filters, and detectors is proprietary to the vendors, so we use approximate models that fit the key properties.

Mirrors
=======

The mirror coatings are protected silver. The model in ``psfsim.mirror_properties`` contains a silver layer coated with a dielectric. The dielectric function of silver is taken from the model of H. Yang et al., *Phys. Rev. B*, **91**:235137 (2015), sample C. We included a single dielectric layer with the index of refraction and thickness chosen to approximate the data provided for public release by L3 Harris (table at 45 degree angle of incidence):

+------------+-------------+------------+-------------+------------+
| Wavelength | S-pol model | S-pol data | P-pol model | P-pol data |
+------------+-------------+------------+-------------+------------+
| 500 nm     | 98.8%       | 99.0%      | 97.5%       | 97.8%      |
+------------+-------------+------------+-------------+------------+
| 1100 nm    | 97.9%       | 97.8%      | 98.2%       | 97.8%      |
+------------+-------------+------------+-------------+------------+
| 2400 nm    | 99.2%       | 99.5%      | 98.7%       | 98.5%      |
+------------+-------------+------------+-------------+------------+

and produce the correct zero-crossing of the linear retardance at ≈ 600 nm. (We also tried 2-layer models, but the reflectance properties for 2-layer models that provide reasonable fits to the data are not significantly different.)

An extremely thin (<< 1 wavelength) binding layer is typically needed to make a durable dielectric layer on silver; we have not attempted to include such a layer in the model.

Filters
=======

**Not yet implemented.**

Detector surface
================

The detector is modeled with a 3-layer dielectric coating on top of HgCdTe bulk with cadmium mole fraction x=0.445.

The dielectric function of the HgCdTe follows:

- The resonance model of A. Djurišić & E. H. Li, *J. Appl. Phys.*, **85**:2854 (1999) at wavelengths <1 µm.
- The exponential model of J. Chu et al., *J. Appl. Phys.*, **75**:1234 (1994) with band gap from G. Hansen et al., *J. Appl. Phys.*, **53**:7099 (1982) at wavelengths >1.5 µm.
- An interpolation in the range 1-1.5 µm.

*The dielectric coating is a toy model right now, we'll fix this soon.*
