# analysis
 Data analysis repo for everything. Created on Feb.2024 with the aim to include all data analysis routine in one repository.
 
 This repository will include all old data analysis functions from Bdot_analysis, LP_analysis, and hairpin_analysis. The aim is to not have duplicate code for each analysis routine that does the same thing. Old functions that work for generic cases will be included, and updated with improvement.
 
 For each experiment performed on a different machine, a new branch will be created accordingly with specific analysis routine.

## Repository Structure

This repository is organized to handle various plasma diagnostics analysis routines:

- **Bdot Analysis**: Processing and visualization of magnetic field measurements
- **Langmuir Probe (LP) Analysis**: Plasma parameter extraction from I-V characteristics
- **Hairpin Analysis**: Electron density measurements using resonant probes

## Recent Updates

### November 2024
- Added X-ray and Bdot data processing routines (`process_xray_bdot.py`)
- Implemented STFT spectrograms for frequency analysis of magnetic fluctuations
- Added photon counting analysis for X-ray diagnostics
- Improved data averaging capabilities for multi-shot experiments

## Usage

Example usage for X-ray and Bdot processing:
