"""data_analysis.plasma — plasma physics.

``formulas`` holds parameter calculations (densities, sound speed, collision
frequencies); ``profiles`` holds radial profile pressure, signed gradients,
floating potential, sheath factor, and local magnetized-plasma scales;
``modes`` holds azimuthal/axial mode decomposition, B-dot dipole quadrature,
and plasma-frame dispersion helpers; ``stability`` holds linear magnetized-plasma
eigenvalue solvers (local/axial Conducting Wall Mode, 5-field Braginskii EM
drift-Alfvén BVP, and 1-D cylindrical radial KH/CWM modes);
``langmuir`` holds Langmuir-probe IV analysis; ``mach`` holds
Mach-probe area-ratio calibration and flow estimators; ``flow`` holds
diagnostic-agnostic in-plane flow-field geometry (polar resolution about a
fitted rotation centre); ``bdot`` holds B-dot probe calibration (area
matrix) and time integration to B; ``fields`` holds diagnostic-agnostic
gridded-field operators (curl / J, Poynting, J.E, Faraday E_phi, vector
potential, cycle averages, AC conductivity); ``photons`` holds
the ``Photons``/``PhotonPulse`` pulse-detection classes; ``cts`` and
``sheffield_thomson`` hold collective Thomson-scattering spectral-density and
form-factor calculations.
"""
