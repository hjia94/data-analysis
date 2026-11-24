# Aug-2026 UCLA LAPD Alfvén Zonal-Flow Campaign

Analysis scripts for the August 10–16, 2026 LAPD campaign. On midas the data are at
`/data/BAPSF_Data/Alfven_zonal_flow/aug2026`: Runs `00`–`36`, LAPD_DAQ (pydaq) HDF5, 254 GB.
Two trombone antennas, south (P9–12) and north (P45–48), launch shear Alfvén waves in
~188 kHz bursts in a 1 kG helium plasma. At each probe position the shots cycle through
**A** (south), **B** (north) and **AB** (both).

Outputs go outside the repo, to `~/data-analysis-output/Aug2026` by default (`--out-dir`).
The scripts use their own default rather than `data_analysis.io.paths.OUTPUT_ROOT`, which
is a Windows path.

## B-dot

| Script | Product | Purpose |
|---|---|---|
| `Aug2026_bdot.py <run>` | `<run>-bdot-phasors.npz` | Per shot: P28 dB/dt → B (`plasma.bdot`, all three baselines); drive-frequency phasors of B, dB/dt and both antenna currents; per-cycle wave power; drift and noise-floor diagnostics. Burst timing and drive frequency are detected from the antenna currents. |
| `Aug2026_bdot_plot.py <run>` | `figures/<run>-bdot-plane.png` | Real-data checks (see its docstring); for planes, per-mode maps of \|B⊥\|, J_z = curl_z B/μ₀, \|div B⊥\|, and the wave envelope. |
| `Aug2026_bicoherence.py <run>` | `<run>-bicoherence.npz`, `figures/<run>-bicoherence.png` | Three-wave coupling (`signal.bispectral`) between low frequencies f₁ (0.5–20 kHz) and the pump side f₂ (168–208 kHz), on raw dB/dt over the steady burst. Covers fixed probe P30 per antenna mode, with a shot-shuffled surrogate that separates real coupling from trigger locking, and per-position maps of moving probe P28. |

B-dot runs: `03`, `05` (coarse planes), `17`, `18` (vertical lines), `21` (horizontal line),
`29` (cross), `30` (12×12 plane, 2 cm), `35` (31×31 plane, 1 cm, 14 GB).

`bdot_scope` channels:
- `C1`–`C3`: P28 Bx/By/Bz, probe C11, X10 amplifier.
- `C4`–`C6`: P30, probe C13, a fixed reference.
- `C7`/`C8`: south/north antenna current, 20 A/V.

### Status and caveats
- **No coil calibration yet.** C11/C13 effective areas are not applied (`AREA = 1.0`), so B
  and J have the right shape and phase but an arbitrary scale. The calibrations are kept
  separately (Google Drive).
- **Use phasors or the `smooth` baseline for the wave.** Integrated B is dominated by a
  ~1.5 kHz mode and small in-burst amplifier offset shifts. See the `integrate_bdot`
  docstring.
- **Probe → chamber rotation is not applied.** Components are in the probe frame. On run 30
  the measured map has div B⊥ / curl_z B ≈ 0.2, versus 1.1 with Bx's sign flipped, so the
  probe's axis signs and the grid are mutually consistent.
- **B_AB ≠ B_A + B_B exactly.** The residual is 12 % on run 30 and 20 % on run 21; antenna
  phase jitter is < 1°, so the drive doesn't explain it. Not yet understood: it could be
  plasma variation or nonlinearity.

Run 30 results (1728 shots, ~40 s with 12 workers):
- **Time integration:** B(f) = Bdot(f)/(iω) to 3e-4.
- **Baselines:** the three agree at f to ≤ 2e-5.
- **Offset drift:** cut from ~1000× to ~20× |B(f)|.
- **Field structure:** a single field-aligned current channel near (x, y) ≈ (−7, 5) cm, with
  a return current around it, the same in all three modes.

## Bicoherence: low-frequency × pump coupling

`Aug2026_bicoherence.py <run>` runs on the steady burst at df = 500 Hz. The f₁ bins at 0.5 and 1 kHz fall inside
the Hann main lobe of DC and are masked, so the f₁ band is effectively 1.5–20 kHz.

The two numbers compared below:
- **Pump-line b²:** the mean of b² over f₁ at f₂ = f_pump. Each run's value is compared with the
  shot-shuffled surrogate, which keeps only the trigger-locked content.
- **Surrogate:** f₁ is taken from a different shot of the same antenna mode, so it keeps everything locked to
  the trigger and removes coupling to the shot's own fluctuation.

| Run | Antennas | P30 pump-line b² (per mode) | Surrogate | Coupled? |
|---|---|---|---|---|
| 17 | 188 / 193 kHz, vertical line | 0.01–0.03 | 0.006–0.025 | Weak or none |
| 18 | 188 kHz, vertical line | 0.01–0.03 | 0.005–0.011 | Weak |
| 21 | 188 kHz, "straight" horizontal line | 0.13–0.36 | 0.014–0.022 | Yes |
| 29 | 188 kHz, "straight" cross | 0.08–0.41 | 0.006–0.013 | Yes |
| 30 | 188 kHz, "straight" 12×12 plane | 0.07–0.40 | 0.001–0.004 | Yes |
| 35 | 188 kHz, 31×31 plane (P30 By only) | 0.26–0.31 | 0.0004–0.0007 | Yes |

Runs 03 and 05 are not covered. Run 03 has no P30 or north-antenna current, and run 05's antenna currents
are on at the record start, so the burst windowing does not apply.

**What the coupled runs show (P30):**
- **Upper and lower sidebands.** The upper sidebands f_pump + f₁ and the lower sidebands f_pump − f₁ are
  phase-coupled to f₁ for every f₁ in the band. They appear in the maps as a vertical line at
  f₂ = f_pump and a diagonal at f₁ + f₂ = f_pump. Single triads reach b² ≈ 0.6–0.8.
- **A discrete mode at 4.5 kHz.** It has its own sidebands at 183.5 and 192.5 kHz.
- **Random phase across shots.** The f₁ and sideband components have shot-to-shot phase
  consistency ≲ 0.2 while the pump has 0.99, so the coupling is not trigger locking.
- **Same in all antenna modes.** It appears in A, B and AB.

This is the signature of the pump being modulated by the low-frequency fluctuations. Bicoherence does
not give the direction: it cannot tell whether the wave is modulated by background fluctuations or
the wave drives the low-frequency response.

**Moving probe P28:**
- **Run 35:** the coupling is concentrated on an arc at r ≈ 14–16 cm from the grid origin, on the +y and +x side.
  - The arc is seen in Bx where it crosses the x axis and in By where it crosses the y axis, i.e. in
    the radial component.
  - Bz is at noise level everywhere.
  - Runs 29 and 30 show the same pattern near y ≈ 13–15 cm.
- **Not instrumental.** On run 30 the coupling does not follow the local pump amplitude: the
  correlation is −0.14 to +0.10. Amplifier intermodulation would follow it.

