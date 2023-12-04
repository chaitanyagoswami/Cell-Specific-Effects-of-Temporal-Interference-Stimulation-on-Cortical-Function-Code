#!/bin/bash
python3 TI_sim_1d.py 2000 2020 500 1200
python3 TI_sim.py 2000 2020 500 1200
python3 plot_TI_sim.py
