"""data_analysis.plasma — plasma physics.

``formulas`` holds parameter calculations (densities, sound speed, collision
frequencies); ``langmuir`` holds Langmuir-probe IV analysis; ``mach`` holds
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
