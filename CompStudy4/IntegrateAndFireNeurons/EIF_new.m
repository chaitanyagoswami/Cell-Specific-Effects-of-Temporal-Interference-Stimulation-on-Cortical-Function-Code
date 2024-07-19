% $$C\frac{dV}{dt}=-g_L(V-V_L)+g_L\Delta_T\exp\left(\frac{V-V_T}{\Delta_T}\right)+I$$
% Input defined at Line 30

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
V_reset = -68;   %mV
V_spike = Vpeak;   %mv
V_T = -59.9;    % mV
Delta = 3.48;   %mV
tau_ref = 1.7;  %ms
%DEFINE INITIAL VALUES AND VECTORS TO HOLD RESULTS
t_vect = 0:dt:t_end; %will hold vector of times
V_vect = zeros(1,length(t_vect)); %initialize the voltage vector
V_plot_vect = zeros(1,length(t_vect)); %pretty version of V_vect to be plotted, that displays a spike whenever voltage reaches threshold
i = 1; % index denoting which element of V is being assigned
V_vect(i)= -65; %first element of V, i.e. value of V at t=0
V_plot_vect(i) = V_vect(i); %if no spike, then just plot the actual voltage V

tmp = -inf;
for i = 2:length(t_vect)
    I = 50*sin(2*pi*2000*t_vect(i)/1000)+50*sin(2*pi*2010*t_vect(i)/1000); % Input set here
%     I=1;
    if (i-tmp)*dt < tau_ref
        V_vect(i) = V_vect(i-1);
    else
        V_vect(i) = V_vect(i-1) + dt*(g_L*(V_L - V_vect(i-1)) + g_L*Delta*exp((V_vect(i-1) - V_T)/Delta) + I)/C;
    end
    if V_vect(i)>=Vpeak
        V_vect(i) = V_reset;
        V_plot_vect(i) = V_spike;
        tmp=i;
    else
        V_plot_vect(i) = V_vect(i);
    end
end

figure,plot(t_vect, V_plot_vect);