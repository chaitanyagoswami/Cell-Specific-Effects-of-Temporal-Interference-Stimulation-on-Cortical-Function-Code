function [T, S] = FitzHughNagumo_OriginalTIStimOptimizedParameters(twidth, freq, amp, Fs)
init = [-1, -0.6];
tspan = 0:1/Fs:twidth;      % ms
options = odeset('MaxStep', 0.05);
[T, S] = ode45(@(t, s)FNModel(t, s, freq, amp), tspan, init, options);

function ds = FNModel(t, s, freq, amp)
ix = sum(diag(amp)*cos(2*pi*freq*t/1000));
%if t>70
%    ix = 0;
%end
ds = zeros(2, 1);
%ds(1) = 2.6*s(1) - 2*s(1)^3/3 +0.2 - s(2) + ix; %2.6*Vgrid-2*Vgrid.^3/3 + 0.2
%ds(2) = 0.08*(s(1) + 2 - 0.8*s(2));


% PARAMETERS I'M TRYING NOW
% 
 ds(1) = s(1) - s(1)^3/3 - s(2) + ix;  %V-dynamics
 ds(2) = 0.08  *  ( s(1) + 1 - 0.01*s(2) ); %W-dynamics


 % BEST PARAMETERS FOR TI
% 
% ds(1) = s(1) - s(1)^3/3 - s(2) + ix;
% ds(2) = 0.08  *  ( s(1) + 1 - 0.01*s(2) ); 
% Frequencies: 1000 and 1020 Hz. Amplitude is 0.24*[7,-7]
 

% PARAMETERS FOR WHICH TI DOES NOT WORK
% 
% ds(1) = s(1) - s(1)^3/3 - s(2) + ix;
% ds(2) = 0.08*(s(1) + 1 - 1*s(2));


% ALSO DOESN'T WORK FOR
%  ds(1) = s(1) - s(1)^3/3 - s(2) + ix;
% ds(2) = 0.08*(s(1) + 2 - 0.01*s(2));

%
 % PARAMETERS THAT ALSO WORK VERY, VERY WELL
% 
% ds(1) = s(1) - s(1)^3/3 - s(2) + ix;
% ds(2) = 0.08*(s(1) + 1 - 0.01*s(2));

 
% PARAMETERS THAT WORK EVEN BETTER
% 
% ds(1) = s(1) - s(1)^3/3 - s(2) + ix;
% ds(2) = 0.08*(s(1) + 1 - 0.08*s(2));

 % PARAMETERS THAT WORK VERY WELL
% 
 %ds(1) = s(1) - s(1)^3/3 - s(2) + ix;
 %ds(2) = 0.08*(s(1) + 0.6 - 0.8*s(2));

 
 