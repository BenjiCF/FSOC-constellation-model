## FSOC Constellation Model

This project implements a **Free-Space Optical Communication (FSOC) constellation model** that simulates optical inter-satellite links (OISLs) between satellites in orbit.  

---

## Overview

The model provides:
- Optical link simulation between satellites using realistic geometric and physical parameters  
- Support for multiple Walker constellation configurations (e.g., Star, Delta)  
- Evaluation of link quality, coverage, and performance under varying orbital conditions  
- Integration with established optical link models (based on Helsdingen’s lasercomm-link-model)

---

## Features

- FSOC link budget computation  
- Dynamic inter-satellite link geometry modeling  
- Orbital configuration control (Walker parameters, altitude, phasing, etc.)  
- Statistical performance evaluation (latency, throughput, BER)  
- Visualization of constellation geometry and link performance  

---

## New Scripts

- constellation_performance_comparison.py  
- constellation_mission.py  
- Networking.py  
- Geometry.py  
- Data_transfer.py

---

## Usage

- **FSOC constellation modeling:**  
  Run constellation_performance_comparison.py with the desired input parameters.  

- **Air-to-space modeling:**  
  Run mission_level.py with appropriate input configurations  
  (refer to *lasercomm-link-model* by Wieger Helsdingen for input structure).

---

## Note

“This project extends the laser-link model developed by Wieger Helsdingen. 
The original model did not include an open-source license. 
Use of the original code is subject to the author’s permission.”

---

## Author

Developed by **Bram Wagemakers**