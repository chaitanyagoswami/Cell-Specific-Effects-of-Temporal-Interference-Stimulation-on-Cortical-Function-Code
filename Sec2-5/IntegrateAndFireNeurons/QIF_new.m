% $$C\frac{dV}{dt}=a(V-V_{rest})(V-V_c)+I$$
% Input defined at Line 29

clear; %clear all variables
% close all; %close all open figures
%DEFINE PARAMETERS
dt = 1/1000; %time step [ms]
t_end = 1200; %total time of run [ms]
t_StimStart = 100; %time to start injecting current [ms]
t_StimEnd = 1000; %time to end injecting current [ms]
C = 1;  %uF/cm2
g_L = 0.1;  %mS/cm2
V_L = -65;  %mV
Vpeak = 20;  %mV
V_reset = -63.8;   %mV
V_spike = Vpeak;   %mv
V_rest = -63.2; %mV
V_c = -56.6;    %mV
a = 0.0144; %mS/(mV*cm2)
%DEFINE INITIAL VALUES AND VECTORS TO HOLD RESULTS
t_vect = 0:dt:t_end; %will hold vector of times
V_vect = zeros(1,length(t_vect)); %initialize the voltage vector
V_plot_vect = zeros(1,length(t_vect)); %pretty version of V_vect to be plotted, that displays a spike whenever voltage reaches threshold
i = 1; % index denoting which element of V is being assigned
V_vect(i)= V_rest; %first element of V, i.e. value of V at t=0
V_plot_vect(i) = V_vect(i); %if no spike, then just plot the actual voltage V

for i = 2:length(t_vect)
    I = 75*sin(2*pi*2000*t_vect(i)/1000)+75*sin(2*pi*2010*t_vect(i)/1000); % Input set here
%     I=1;
    V_vect(i) = V_vect(i-1) + dt*(a*(V_vect(i-1) - V_rest)*(V_vect(i-1) - V_c) + I)/C;
    if V_vect(i)>=Vpeak
        V_vect(i) = V_reset;
        V_plot_vect(i) = V_spike;
    else
        V_plot_vect(i) = V_vect(i);
    end
end

figure,plot(t_vect, V_plot_vect);