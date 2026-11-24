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
