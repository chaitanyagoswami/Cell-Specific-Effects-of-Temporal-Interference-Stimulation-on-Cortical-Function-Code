Code for running computational studies presented in Section 2.4 of the work "Cell Specific Effects of Temporal Interference Stimulation on Cortical Function".

For running the code: 

1. Unzip the cells.zip and mechansims.zip files.
3. Run the following commmand in the terminal: sh init_setup_run_once.sh
4. To reproduce the results of Sec 2.4.1 run the following command in the terminal: sh run_pointElec_sim.sh
5. To reproduce the results of Sec 2.4.2 run the following command in the terminal: sh run_non-invasive_sim.sh
6. To reproduce the results of Sec 2.4.3 run the following command in the terminal: sh run_TI_sim.sh

Common Issues: 
1. pyshtools is known to cause issues in Windows.
2. To run the simulations you need the NEURON software. Installing the NEURON software can be tricky. Please refer to their setup guide: https://www.neuron.yale.edu/neuron/download, in case init_setup_run_once.sh is not able to install neuron using pip
