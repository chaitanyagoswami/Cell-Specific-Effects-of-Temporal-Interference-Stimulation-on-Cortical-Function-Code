#!/bin/bash
python3 TI_sim_1d.py 2000 2020 500 1200 
python3 TI_sim.py 2000 2020 500 1200 0 # Target
python3 TI_sim.py 2000 2020 500 1200 1 # Non-Target
python3 plot_TI_sim.py
