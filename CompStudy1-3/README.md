Code for running computational studies presented in the work "Cell Specific Effects of Temporal Interference Stimulation on Cortical Function".

For running the code: 

1. Unzip the cells.zip and mechansims.zip files.
3. Run the following commmand in the terminal: sh init_setup_run_once.sh
4. To reproduce the results of Computational Study 1 run the following command in the terminal: sh run_pointElec_sim.sh
5. To reproduce the results of Computational Study 2 run the following command in the terminal: sh run_non-invasive_sim.sh
6. To reproduce the results of Computational Study 3 run the following command in the terminal: sh run_TI_sim.sh

Common Issues: 
1. pyshtools is known to cause issues in Windows.
2. To run the simulations you need the NEURON software. Installing the NEURON software can be tricky. Please refer to their setup guide: https://www.neuron.yale.edu/neuron/download, in case init_setup_run_once.sh is not able to install neuron using pip
3. The code uses the ray library as a parallelization tool to speed up computation, adjust the NUM_CORES parameter according to the parallelization available. If the NUM_CORES is too big, the code will throw an out-of-memory error. The default is set to either 30 or 50 cores.
4. Note that this code uses modified version of the files provided by Aberra et al., so the code will not run with the default files from Aberra et al. The modification are in the xtra mechanism to simulate TI stimulation and the cellChooser.hoc to add L23 Basket Cells.
