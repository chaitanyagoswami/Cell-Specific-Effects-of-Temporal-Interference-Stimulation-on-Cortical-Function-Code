#!/bin/bash
python3 non-invasive_sim.py 0 100 1000
python3 non-invasive_sim.py 1 100 1000
python3 non-invasive_sim.py 2 100 1000
python3 plot_summary_non-invasive.py 0
python3 plot_summary_non-invasive.py 1
python3 plot_summary_non-invasive.py 2
