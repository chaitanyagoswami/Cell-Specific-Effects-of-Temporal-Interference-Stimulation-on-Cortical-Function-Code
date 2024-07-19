clear all;
%close all;
% Showing that PV neurons do not exhibit TI properties
% High frequency case

disp('0.29 * [307, -307] and 3000;3080 Hz frequencies achieve TI with square waves');
disp('0.1*[307; -307] and 800; 820 Hz leads to small amplitude current imbalances that add up');
freq = [1600;1633];
amp1 = 0.61*[307; -307];  % min. amp. such that firing rate=beating freq.
amp2 = 0.61*[613; 1];   % bias towards one direction, keeping total amp same
Fs = 80; % kHz
[T,V1,m1,n1,h1]=HodgkinHuxleyFast_edited(-70,1000,freq,amp1,Fs);
Iinput = sum(diag(amp1)*cos(2*pi*freq*T'/1000))';% Just for plotting
%Iinput = Iinput.*(Iinput>0); 

[b,a]=butter(3,[100/(Fs*500), 1000/(Fs*500)]);
Vfilt = filtfilt(b,a,V1);

%y = lowpass(V1,2000,Fs*1000);


figure;
set(gcf, 'pos', [340, 15, 1220, 960]);
subplot(5,1,1), plot(T, Iinput, 'b', 'LineWidth', 1.5);
ylabel('Input current')%, title('Freq. [4000, 4015]')
ax = gca;
set(ax, 'FontSize', 22);
%ax.XAxis.Visible = 'off';
xlim([100, 180])
%ylim([-700,700])
set(gca,'FontSize',22);

subplot(5,1,2), plot(T, V1, 'r', 'LineWidth', 1.5);

xlabel('Time (ms)'), ylabel({'Membrane','potential (mV)'})
set(gca,'FontSize',18);
xlim([100, 180])
%ylim([-120,70])

subplot(5,1,3), %plot(T, m1.^3.*h1, 'r', 'LineWidth', 1.5);
plot(T,Vfilt,'k','LineWidth',1.5);
%hold;
%plot(T,y,'r')
xlabel('Time (ms)'), ylabel({'Filtered' ,'mem. potential'})
set(gca,'FontSize',18);
xlim([100, 180])
%ylim([0,1])
%ylim([-120,70])

subplot(5,1,4), plot(T, m1, 'r', 'LineWidth', 1.5);
xlabel('Time (ms)'), ylabel('m')
set(gca,'FontSize',20);
%hold;
%dif = diff(m1)./max(diff(m1));
%plot(T,[0, dif']);
xlim([100, 180])
ylim([0,1])

C = 1.0;
gNabar = 120.0;
gKbar = 36.0;
gLbar = 0.3;
ENa = 45.0;
EK = -82.0;
EL = -59.0;



cycle_length = Fs*1000/freq(1);
NoOfCycles = 300*freq(1)/1000;
IntNaCurrent = zeros(length(T),1); IntKCurrent=IntNaCurrent; IntLeakCurrent = IntNaCurrent;
dt = 1/Fs/1000;
NaCurrent = gNabar*m1.^3.*h1.*(V1-ENa);
LeakCurrent = gLbar*(V1-EL);
KCurrent = gKbar*n1.^4.*(V1-EK);

for i=1:NoOfCycles
    IntNaCurrent(cycle_length*(i-1)+1:cycle_length*i) = sum(NaCurrent(cycle_length*(i-1)+1:cycle_length*i))*dt;
    IntKCurrent(cycle_length*(i-1)+1:cycle_length*i) = sum(KCurrent(cycle_length*(i-1)+1:cycle_length*i))*dt;
    IntLeakCurrent(cycle_length*(i-1)+1:cycle_length*i) = sum(LeakCurrent(cycle_length*(i-1)+1:cycle_length*i))*dt;
%   IntW(cycle_length*(i-1)+1:cycle_length*i) = sum(W(cycle_length*(i-1)+1:cycle_length*i))*dt;
end

%IntNaCurrent=[IntNaCurrent' IntNaCurrent(length(IntNaCurrent))]';
%IntW=[IntW' IntW(length(IntW))]';




subplot(5,1,5), plot(T, IntNaCurrent, 'b', 'LineWidth', 1.5);
%plot(T, gLbar*(V2-EL), 'k', 'LineWidth', 1.5);
hold;
plot(T, IntKCurrent, 'r', 'LineWidth', 1.5);
plot(T, IntLeakCurrent, 'k', 'LineWidth', 1.5);
plot(T, IntLeakCurrent+IntKCurrent+IntNaCurrent, 'r', 'LineWidth', 1);
%plot(T, NaCurrent, 'k', 'LineWidth', 1.5);
%plot(T, gNabar*m2.^3.*h2.*(V2-ENa), 'r', 'LineWidth', 1.5);
%plot(T, gKbar*n2.^4.*(V2-EK), 'b', 'LineWidth', 1.5);
%plot(T, gNabar*m2.^3.*h2.*(V2-ENa)+ gKbar*n2.^4.*(V2-EK) + gLbar*(V2-EL), 'c', 'LineWidth', 1.5);
xlabel('Time (ms)'), ylabel({'Currents integrated','over one cycle'})
set(gca,'FontSize',22);
xlim([100, 180])
%ylim([0,1])
%legend('Leakage','Sodium','Potassium');
legend('Sodium','Potassium','Leakage','Total');


%subplot(5,1,5), plot(T, n1, 'r', 'LineWidth', 1.5);
%xlabel('Time (ms)'), ylabel('n')
%set(gca,'FontSize',25);
%xlim([100, 180])
%ylim([0,1])



savefig('BadTI/freq4000_center')
saveas(gcf, 'BadTI/freq4000_center.png')

figure;
set(gcf, 'pos', [340, 15, 1220, 960]);
[T,V2,m2,n2,h2]=HodgkinHuxleyFast_edited(-70,1000,freq,amp2,Fs);
Iinput = sum(diag(amp2)*cos(2*pi*freq*T'/1000))';
%Iinput = Iinput.*(Iinput>0);


subplot(5,1,1), plot(T, Iinput, 'b','LineWidth', 1.5);
ylabel('Input current')%, title('Freq. [4000, 4015]')
ax = gca;
set(ax, 'FontSize', 22);
%ax.XAxis.Visible = 'off';
xlim([100, 180])
%ylim([-700,700])
set(gca,'FontSize',22);


subplot(5,1,2), plot(T, V2, 'r','LineWidth', 1.5);
xlabel('Time (ms)'), ylabel({'Membrane','potential (mV)'})
set(gca,'FontSize',18);
xlim([100, 180])
ylim([-200,200])

[b,a]=butter(3,[100/(Fs*500), 1000/(Fs*500)]);
Vfilt2 = filtfilt(b,a,V2);



subplot(5,1,3), %plot(T, m1.^3.*h1, 'r', 'LineWidth', 1.5);
plot(T,Vfilt2,'k','LineWidth',1.5);
xlabel('Time (ms)'), ylabel({'Filtered' ,'mem. potential'})
set(gca,'FontSize',18);
xlim([100, 180])
ylim([-100,100])


subplot(5,1,4), plot(T, m2, 'r','LineWidth', 1.5);
xlabel('Time (ms)'), ylabel('m')
set(gca,'FontSize',22);
%hold;
%dif = diff(m2)./max(diff(m2));
%plot(T,[0, dif']);
xlim([100, 180])
ylim([0,1])



cycle_length = Fs*1000/freq(1);
NoOfCycles = 300*freq(1)/1000;
IntNaCurrent = zeros(length(T),1); IntKCurrent=IntNaCurrent; IntLeakCurrent = IntNaCurrent;
dt = 1/Fs/1000;
NaCurrent = gNabar*m2.^3.*h2.*(V2-ENa);
LeakCurrent = gLbar*(V2-EL);
KCurrent = gKbar*n2.^4.*(V2-EK);

for i=1:NoOfCycles
    IntNaCurrent(cycle_length*(i-1)+1:cycle_length*i) = sum(NaCurrent(cycle_length*(i-1)+1:cycle_length*i))*dt;
    IntKCurrent(cycle_length*(i-1)+1:cycle_length*i) = sum(KCurrent(cycle_length*(i-1)+1:cycle_length*i))*dt;
    IntLeakCurrent(cycle_length*(i-1)+1:cycle_length*i) = sum(LeakCurrent(cycle_length*(i-1)+1:cycle_length*i))*dt;
%   IntW(cycle_length*(i-1)+1:cycle_length*i) = sum(W(cycle_length*(i-1)+1:cycle_length*i))*dt;
end

%IntNaCurrent=[IntNaCurrent' IntNaCurrent(length(IntNaCurrent))]';
%IntW=[IntW' IntW(length(IntW))]';




subplot(5,1,5), plot(T, IntNaCurrent, 'b', 'LineWidth', 1.5);
%plot(T, gLbar*(V2-EL), 'k', 'LineWidth', 1.5);
hold;
plot(T, IntKCurrent, 'r', 'LineWidth', 1.5);
plot(T, IntLeakCurrent, 'k', 'LineWidth', 1.5);
plot(T, IntLeakCurrent+IntKCurrent+IntNaCurrent, 'g--', 'LineWidth', 2.5);
%plot(T, NaCurrent, 'k', 'LineWidth', 1.5);
%plot(T, gNabar*m2.^3.*h2.*(V2-ENa), 'r', 'LineWidth', 1.5);
%plot(T, gKbar*n2.^4.*(V2-EK), 'b', 'LineWidth', 1.5);
%plot(T, gNabar*m2.^3.*h2.*(V2-ENa)+ gKbar*n2.^4.*(V2-EK) + gLbar*(V2-EL), 'c', 'LineWidth', 1.5);
xlabel('Time (ms)'), ylabel({'Currents integrated','over one cycle'})
set(gca,'FontSize',22);
xlim([100, 180])
%ylim([0,1])
legend('Sodium','Potassium','Leakage','Total');


% 
% 
% subplot(5,1,5), plot(T, n2, 'r');
% xlabel('Time (ms)'), ylabel('n')
% set(gca,'FontSize',25);
% xlim([100, 180])
% ylim([0,1])





savefig('BadTI/freq4000_offcenter')
saveas(gcf, 'BadTI/freq4000_offcenter.png')

% % Same frequency as in STIMULUS, but with PV
% freq = [2500; 2515];
% amp1 = [180; 180];  % min. amp. such that firing rate=beating freq.
% amp2 = [350; 10];   % bias towards one direction, keeping total amp same
% Fs = 50; % kHz
% [T,V]=HodgkinHuxleyPV_fast(-70,1000,freq,amp1,Fs);
% Iinput = sum(diag(amp1)*cos(2*pi*freq*T'/1000))';
% 
% figure;
% set(gcf, 'pos', [340, 15, 1220, 960]);
% subplot(2,1,1), plot(T, Iinput, 'k');
% ylabel('Input current'), title('Freq. [2500, 2515]')
% ax = gca;
% set(ax, 'FontSize', 18);
% %ax.XAxis.Visible = 'off';
% xlim([100, 900])
% ylim([-700,700])
% 
% subplot(2,1,2), plot(T, V, 'k');
% xlabel('Time (ms)'), ylabel('Membrane potential (mV)')
% set(gca, 'FontSize', 18);
% xlim([100, 900])
% ylim([-120,70])
% 
% savefig('BadTI/freq2500_center')
% saveas(gcf, 'BadTI/freq2500_center.png')
% 
% figure;
% set(gcf, 'pos', [340, 15, 1220, 960]);
% [T,V]=HodgkinHuxleyPV_fast(-70,1000,freq,amp2,Fs);
% Iinput = sum(diag(amp2)*cos(2*pi*freq*T'/1000))';
% subplot(2,1,1), plot(T, Iinput, 'k');
% ylabel('Input current'), title('Freq. [2500, 2515]')
% ax = gca;
% set(ax, 'FontSize', 18);
% %ax.XAxis.Visible = 'off';
% xlim([100, 900])
% ylim([-700,700])
% 
% subplot(2,1,2), plot(T, V, 'k');
% xlabel('Time (ms)'), ylabel('Membrane potential (mV)')
% set(gca, 'FontSize', 18);
% xlim([100, 900])
% ylim([-120,70])
% 
% savefig('BadTI/freq2500_offcenter')
% saveas(gcf, 'BadTI/freq2500_offcenter.png')
