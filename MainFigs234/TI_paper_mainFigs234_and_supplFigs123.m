%% TI_paper_mainFigs234_and_supplFigs123.mat %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Sara Caldas-Martinez
% Last checked 240720

% COMMENTS:
% -

clear all;
close all;
clc;


%% Fig1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Abstract pure and modulated sinusoids that we use in main Figures 1, 2,
% 3, and 4; and SI Figures 1, 2, and 3.

% Pure sinusoid
ff=80; % frequency [Hz]
tt=(0:1/(ff*10):1);
aa=1; % amplitude [V]
phi=0; % phase
yy=aa*sin(2*pi*ff*tt+phi);

fig1a = figure;
plot(tt,yy, 'LineWidth', 2, 'color', 'black');
xlim([0 0.5]);
ylim([-5 5]);
box off;
axis off;
saveas(gcf,'pureSinusoid.png')

% Modulated sinusoid
fTI=100; %frequency [Hz]
tTI=(0:1/(ff*10):1);
aTI=0.5; %amplitude [V]
phi=0; %phase
yTI=-(aTI*sin(2*pi*fTI*tTI+phi))+aTI*sin(2*pi*ff*tt+phi);

fig1b = figure;
plot(tTI,yTI, 'LineWidth', 2, 'color', 'black');
xlim([0 0.5]);
ylim([-5 5])
box off;
axis off;
saveas(gcf,'modulatedSinusoid.png')


%% Fig2 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Pyr exhibits TI
% 20210409a

% 1 - Extract data from txt file
mainFolder = cd('C:\Users\chait\OneDrive\Desktop\SARA\specialCase\recheckingAll_240720\inputFiles'); % CHANGE directory accordingly where the inputFiles folder is
% ephys file
part1Rec = readtable("SCM_20210409a_1aaCL2-3_sePtIr4f_{0}.txt");
part2Rec = readtable("SCM_20210409a_1aaCL2-3_sePtIr4f_{1}.txt");
% notes file
notes = readtable('20210409a_Notes.txt');
selectedIntraSweep = 16
selectedSinSweep = (table2array(notes(17,1)))+1
selectedTISweep = (table2array(notes(9,1)))+2
cd(mainFolder); %% return to main directory that should be where the outputFiles folder is

% 2 - Prepare data
newnames = matlab.lang.makeUniqueStrings([part1Rec.Properties.VariableNames, part2Rec.Properties.VariableNames]);
part1Rec.Properties.VariableNames = newnames(1:numel(part1Rec.Properties.VariableNames));
part2Rec.Properties.VariableNames = newnames(numel(part1Rec.Properties.VariableNames)+1:end);
Rec = [part1Rec part2Rec];
% remove NaN columns
badColumns = isnan(Rec{1,:});
CleanRec = Rec(:,~badColumns);
% segregate Time and Voltage sweeps
Time_Sweeps = CleanRec(:,1:2:end-1);
Voltage_Sweeps = CleanRec(:,2:2:end);

% 3 - Plot traces
% firing pattern --> intracellular stimulation
currentVoltage_Sweep = table2array(Voltage_Sweeps(:,selectedIntraSweep));
currentTime_Sweep = table2array(Time_Sweeps(:,selectedIntraSweep));
[b,a] = butter(3,[100,1000]/(10000/2));
filtVoltage_Sweep = filtfilt(b,a,currentVoltage_Sweep);
[pks,locs] = findpeaks(filtVoltage_Sweep, currentTime_Sweep, 'MinPeakProminence',20, "MinPeakHeight",15);
NumPeaks_Sweeps = length(pks);

Fig1c1 = figure;
x = currentTime_Sweep;
y1 = currentVoltage_Sweep;
plot(x,y1,'black','LineWidth',1);
title({'PYR', 'Intra Stim','exhibits TI'});
xlim([[1.8 2.7]]);
ylim([-130 100]);
xlabel('Time (s)', 'FontSize', 14);
ylabel('Voltage (mV)', 'FontSize', 14);
set(gca,'FontSize',14);
saveas(gcf,'Fig2_20210409a_PYR_IntraStim_exhibitsTI_scale.png');

currentVoltage_Sweep = table2array(Voltage_Sweeps(:,selectedSinSweep));
currentTime_Sweep = table2array(Time_Sweeps(:,selectedSinSweep));
[b,a] = butter(3,[100,1000]/(10000/2));
filtVoltage_Sweep = filtfilt(b,a,currentVoltage_Sweep);
[pks,locs] = findpeaks(filtVoltage_Sweep, currentTime_Sweep, 'MinPeakProminence',20, "MinPeakHeight",15);
NumPeaks_Sweeps = length(pks);

Fig1d1 = figure;
x = currentTime_Sweep;
y1 = currentVoltage_Sweep;
plot(x,y1,'black','LineWidth',1);
title({'PYR', 'Sin Stim','exhibits TI'});
x1 = (table2array(notes(20,1))) - 0.2
x2 = x1 + 0.9
xlim([x1 x2]);
ylim([-130 100]);
xlabel('Time (s)', 'FontSize', 14);
ylabel('Voltage (mV)', 'FontSize', 14);
set(gca,'FontSize',14);
saveas(gcf,'Fig2_20210409a_PYR_SinStim_exhibitsTI_scale.png');

Fig1d2 = figure;
x = currentTime_Sweep;
y1 = filtVoltage_Sweep;
plot(x,y1,'black','LineWidth',1);
title({'PYR', 'Sin Stim Filt','exhibits TI'});
x1 = (table2array(notes(20,1))) - 0.2
x2 = x1 + 0.9
xlim([x1 x2]);
ylim([-130 100]);
xlabel('Time (s)', 'FontSize', 14);
ylabel('Voltage (mV)', 'FontSize', 14);
box off;
axis off;
saveas(gcf,'Fig2_20210409a_PYR_SinStimFilt_exhibitsTI.png');

% Fig1d3 = figure;
% x = currentTime_Sweep;
% y1 = currentVoltage_Sweep;
% plot(x,y1,'black','LineWidth',1);
% title({'PYR', 'Sin Stim Zoom','exhibits TI'});
% xlim([6.22 6.24]);
% ylim([-130 100]);
% xlabel('Time (s)', 'FontSize', 14);
% ylabel('Voltage (mV)', 'FontSize', 14);
% set(gca,'FontSize',14);
% % box off;
% % axis off;
% saveas(gcf,'Fig2_20210409a_PYR_SinStimZoom_exhibitsTI.png');

Fig1d4 = figure;
x = currentTime_Sweep;
y1 = filtVoltage_Sweep;
plot(x,y1,'black','LineWidth',1);
title({'PYR', 'Sin Stim Zoom Filt','exhibits TI'});
xlim([6.22 6.24]);
ylim([-130 100]);
xlabel('Time (s)', 'FontSize', 14);
ylabel('Voltage (mV)', 'FontSize', 14);
set(gca,'FontSize',14);
saveas(gcf,'Fig2_20210409a_PYR_SinStimZoomFilt_exhibitsTI_scale.png');

currentVoltage_Sweep = table2array(Voltage_Sweeps(:,selectedTISweep));
currentTime_Sweep = table2array(Time_Sweeps(:,selectedTISweep));
[b,a] = butter(3,[100,1000]/(10000/2));
filtVoltage_Sweep = filtfilt(b,a,currentVoltage_Sweep);
[pks,locs] = findpeaks(filtVoltage_Sweep, currentTime_Sweep, 'MinPeakProminence',20, "MinPeakHeight",15);
NumPeaks_Sweeps = length(pks);

Fig1e1 = figure;
x = currentTime_Sweep;
y1 = currentVoltage_Sweep;
plot(x,y1,'black','LineWidth',1);
title({'PYR', 'TI Stim','exhibits TI'});
x1 = (table2array(notes(13,1))) - 0.2
x2 = x1 + 0.9
xlim([x1 x2]);
ylim([-130 100]);
xlabel('Time (s)', 'FontSize', 14);
ylabel('Voltage (mV)', 'FontSize', 14);
set(gca,'FontSize',14);
box off;
axis off;
saveas(gcf,'Fig2_20210409a_PYR_TIStim_exhibitsTI.png');

Fig1e2 = figure;
x = currentTime_Sweep;
y1 = filtVoltage_Sweep;
plot(x,y1,'black','LineWidth',1);
title({'PYR', 'TI Stim Filt','exhibits TI'});
x1 = (table2array(notes(13,1))) - 0.2
x2 = x1 + 0.9
xlim([x1 x2]);
ylim([-130 100]);
xlabel('Time (s)', 'FontSize', 14);
ylabel('Voltage (mV)', 'FontSize', 14);
set(gca,'FontSize',14);
box off;
axis off;
saveas(gcf,'Fig2_20210409a_PYR_TIStimFilt_exhibitsTI.png');

% Fig1e3 = figure;
% x = currentTime_Sweep;
% y1 = currentVoltage_Sweep;
% plot(x,y1,'black','LineWidth',1);
% title({'PYR', 'TI Stim Zoom','exhibits TI'});
% xlim([6.12 6.14]);
% ylim([-130 100]);
% xlabel('Time (s)', 'FontSize', 14);
% ylabel('Voltage (mV)', 'FontSize', 14);
% set(gca,'FontSize',14);
% % box off;
% % axis off;
% saveas(gcf,'Fig2_20210409a_PYR_TIStimZoom_exhibitsTI.png');

Fig1e4 = figure;
x = currentTime_Sweep;
y1 = filtVoltage_Sweep;
plot(x,y1,'black','LineWidth',1);
title({'PYR', 'TI Stim Zoom Filt','exhibits TI'});
xlim([6.1465 6.1665]);
ylim([-130 100]);
xlabel('Time (s)', 'FontSize', 14);
ylabel('Voltage (mV)', 'FontSize', 14);
set(gca,'FontSize',14);
box off;
axis off;
saveas(gcf,'Fig2_20210409a_PYR_TIStimZoomFilt_exhibitsTI.png');


%% PV does not exhibits TI
% 20210807b

% 1 - Extract data from txt file
mainFolder = cd('C:\Users\chait\OneDrive\Desktop\SARA\specialCase\recheckingAll_240720\inputFiles'); % CHANGE directory accordingly where the inputFiles folder is
% ephys file
part1Rec = readtable("SCM_20210807b_3aaCL2-3_seIr-Pt2f_{0}.txt"); %%15,15
part2Rec = readtable("SCM_20210807b_3aaCL2-3_seIr-Pt2f_{1}.txt");
% notes file
notes = readtable('20210807b_Notes.txt');
selectedIntraSweep = 16
selectedSinSweep = (table2array(notes(17,1)))
selectedTISweep = (table2array(notes(9,1))) + 1
cd(mainFolder); %% return to main directory that should be where the outputFiles folder is

% 2 - Prepare data
newnames = matlab.lang.makeUniqueStrings([part1Rec.Properties.VariableNames, part2Rec.Properties.VariableNames]);
part1Rec.Properties.VariableNames = newnames(1:numel(part1Rec.Properties.VariableNames));
part2Rec.Properties.VariableNames = newnames(numel(part1Rec.Properties.VariableNames)+1:end);
Rec = [part1Rec part2Rec];
% remove NaN columns
badColumns = isnan(Rec{1,:});
CleanRec = Rec(:,~badColumns);
% segregate Time and Voltage sweeps
Time_Sweeps = CleanRec(:,1:2:end-1);
Voltage_Sweeps = CleanRec(:,2:2:end);

% 3 - Plot traces
% firing pattern --> intracellular stimulation
currentVoltage_Sweep = table2array(Voltage_Sweeps(:,selectedIntraSweep));
currentTime_Sweep = table2array(Time_Sweeps(:,selectedIntraSweep));
[b,a] = butter(3,[100,1000]/(10000/2));
filtVoltage_Sweep = filtfilt(b,a,currentVoltage_Sweep);
[pks,locs] = findpeaks(filtVoltage_Sweep, currentTime_Sweep, 'MinPeakProminence',20, "MinPeakHeight",15);
NumPeaks_Sweeps = length(pks);

Fig1c1 = figure;
x = currentTime_Sweep;
y1 = currentVoltage_Sweep;
plot(x,y1,'black','LineWidth',1);
title({'PV', 'Intra Stim','exhibits TI'});
xlim([[1.8 2.7]]);
ylim([-130 100]);
xlabel('Time (s)', 'FontSize', 14);
ylabel('Voltage (mV)', 'FontSize', 14);
set(gca,'FontSize',14);
box off;
axis off;
saveas(gcf,'Fig2_20210807b_PV_IntraStim_exhibitsTI.png');

% Fig1c2 = figure;
% x = currentTime_Sweep;
% y1 = filtVoltage_Sweep;
% plot(x,y1,'black','LineWidth',1);
% title({'PV', 'Intra Stim Filt','exhibits TI'});
% xlim([[1.8 2.7]]);
% ylim([-130 100]);
% xlabel('Time (s)', 'FontSize', 14);
% ylabel('Voltage (mV)', 'FontSize', 14);
% set(gca,'FontSize',14);
% % box off;
% % axis off;
% saveas(gcf,'Fig2_20210807b_PV_IntraStimFilt_exhibitsTI.png');

currentVoltage_Sweep = table2array(Voltage_Sweeps(:,selectedSinSweep));
currentTime_Sweep = table2array(Time_Sweeps(:,selectedSinSweep));
[b,a] = butter(3,[100,1000]/(10000/2));
filtVoltage_Sweep = filtfilt(b,a,currentVoltage_Sweep);
[pks,locs] = findpeaks(filtVoltage_Sweep, currentTime_Sweep, 'MinPeakProminence',20, "MinPeakHeight",15);
NumPeaks_Sweeps = length(pks);

Fig1d1 = figure;
x = currentTime_Sweep;
y1 = currentVoltage_Sweep;
plot(x,y1,'black','LineWidth',1);
title({'PV', 'Sin Stim','exhibits TI'});
x1 = (table2array(notes(19,1))) - 0.2
x2 = x1 + 0.9
xlim([x1 x2]);
ylim([-130 100]);
xlabel('Time (s)', 'FontSize', 14);
ylabel('Voltage (mV)', 'FontSize', 14);
set(gca,'FontSize',14);
box off;
axis off;
saveas(gcf,'Fig2_20210807b_PV_SinStim_exhibitsTI.png');

Fig1d2 = figure;
x = currentTime_Sweep;
y1 = filtVoltage_Sweep;
plot(x,y1,'black','LineWidth',1);
title({'PV', 'Sin Stim Filt','exhibits TI'});
x1 = (table2array(notes(19,1))) - 0.2
x2 = x1 + 0.9
xlim([x1 x2]);
ylim([-130 100]);
xlabel('Time (s)', 'FontSize', 14);
ylabel('Voltage (mV)', 'FontSize', 14);
box off;
axis off;
saveas(gcf,'Fig2_20210807b_PV_SinStimFilt_exhibitsTI.png');

Fig1d4 = figure;
x = currentTime_Sweep;
y1 = filtVoltage_Sweep;
plot(x,y1,'black','LineWidth',1);
title({'PV', 'Sin Stim Zoom Filt','exhibits TI'});
xlim([6.85 6.87]);
ylim([-130 100]);
xlabel('Time (s)', 'FontSize', 14);
ylabel('Voltage (mV)', 'FontSize', 14);
set(gca,'FontSize',14);
box off;
axis off;
saveas(gcf,'Fig2_20210807b_PV_SinStimZoomFilt_exhibitsTI.png');

currentVoltage_Sweep = table2array(Voltage_Sweeps(:,selectedTISweep));
currentTime_Sweep = table2array(Time_Sweeps(:,selectedTISweep));
[b,a] = butter(3,[100,1000]/(10000/2));
filtVoltage_Sweep = filtfilt(b,a,currentVoltage_Sweep);
[pks,locs] = findpeaks(filtVoltage_Sweep, currentTime_Sweep, 'MinPeakProminence',20, "MinPeakHeight",15);
NumPeaks_Sweeps = length(pks);

Fig1e1 = figure;
x = currentTime_Sweep;
y1 = currentVoltage_Sweep;
plot(x,y1,'black','LineWidth',1);
title({'PV', 'TI Stim','exhibits TI'});
x1 = (table2array(notes(12,1))) - 0.2
x2 = x1 + 0.9
xlim([x1 x2]);
ylim([-130 100]);
xlabel('Time (s)', 'FontSize', 14);
ylabel('Voltage (mV)', 'FontSize', 14);
set(gca,'FontSize',14);
box off;
axis off;
saveas(gcf,'Fig2_20210807b_PV_TIStim_exhibitsTI.png');

Fig1e2 = figure;
x = currentTime_Sweep;
y1 = filtVoltage_Sweep;
plot(x,y1,'black','LineWidth',1);
title({'PV', 'TI Stim Filt','exhibits TI'});
x1 = (table2array(notes(12,1))) - 0.2
x2 = x1 + 0.9
xlim([x1 x2]);
ylim([-130 100]);
xlabel('Time (s)', 'FontSize', 14);
ylabel('Voltage (mV)', 'FontSize', 14);
set(gca,'FontSize',14);
box off;
axis off;
saveas(gcf,'Fig2_20210807b_PV_TIStimFilt_exhibitsTI.png');

Fig1e4 = figure;
x = currentTime_Sweep;
y1 = filtVoltage_Sweep;
plot(x,y1,'black','LineWidth',1);
title({'PV', 'TI Stim Zoom Filt','exhibits TI'});
xlim([6.83 6.85]);
ylim([-130 100]);
xlabel('Time (s)', 'FontSize', 14);
ylabel('Voltage (mV)', 'FontSize', 14);
set(gca,'FontSize',14);
box off;
axis off;
saveas(gcf,'Fig2_20210807b_PV_TIStimZoomFilt_exhibitsTI.png');


%% Does or does not exhibit TI PIE CHARTS
mainFolder = cd('C:\Users\chait\OneDrive\Desktop\SARA\specialCase\recheckingAll_240720\inputFiles'); % CHANGE directory accordingly where the inputFiles folder is
results = readtable('allCellsResultsSummary.xlsx');
cd(mainFolder); %% return to main directory that should be where the outputFiles folder is

Pyr_yesTI = 0;
Pyr_noTI = 0;
Pyr_total = 0;

PV_yesTI = 0;
PV_noTI = 0;
PV_total = 0;

for i = 1:height(results);
    if string(results{i,2}) == 'Pyr'
        Pyr_total = Pyr_total + 1;
        if string(results{i,8}) == 'beforeDrugs'
            if string(results{i,47}) == 'yes'
                Pyr_yesTI = Pyr_yesTI + 1;
            end
        else
            if string(results{i,47}) == 'yes' || string(results{i,48}) == 'yes'
                Pyr_yesTI = Pyr_yesTI + 1;
            end
        end
        Pyr_noTI = Pyr_total - Pyr_yesTI;
    end
    if string(results{i,2}) == 'PV'
        PV_total = PV_total + 1;
        if string(results{i,8}) == 'beforeDrugs'
            if string(results{i,47}) == 'yes'
                PV_yesTI = PV_yesTI + 1;
            end
        else
            if string(results{i,47}) == 'yes' || string(results{i,48}) == 'yes'
                PV_yesTI = PV_yesTI + 1;
            end
        end
        PV_noTI = PV_total - PV_yesTI;
    end
end

Fig1f1 = figure;
Pyr = [Pyr_yesTI Pyr_noTI];
h = pie(Pyr);
set(findobj(gca,'type','line'),'linew',2);
% title({'Pyr'});
% set(gca,'FontSize',20);
% set(findobj(h,'type','text'),'fontsize',20);
colormap([1 1 1;0 0 0]);
saveas(gcf,'Fig2_OverallPyr_Pie.png');

Fig1f2 = figure;
PV = [PV_yesTI PV_noTI];
h = pie(PV);
set(findobj(gca,'type','line'),'linew',2);
% title({'PV'});
% set(gca,'FontSize',20);
% set(findobj(h,'type','text'),'fontsize',20);
colormap([1 1 1;0 0 0]);
saveas(gcf,'Fig2_OverallPV_Pie.png');


%% Fig3 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Each Frequency Pies
mainFolder = cd('C:\Users\chait\OneDrive\Desktop\SARA\specialCase\recheckingAll_240720\inputFiles'); % CHANGE directory accordingly where the inputFiles folder is
results = readtable('allCellsResultsSummary.xlsx');
cd(mainFolder); %% return to main directory that should be where the outputFiles folder is

Pyr_yesTI_2kHz = 0;
Pyr_noTI_2kHz = 0;
Pyr_total_2kHz = 0;

Pyr_yesTI_4kHz = 0;
Pyr_noTI_4kHz = 0;
Pyr_total_4kHz = 0;

PV_yesTI_2kHz = 0;
PV_noTI_2kHz = 0;
PV_total_2kHz = 0;

PV_yesTI_4kHz = 0;
PV_noTI_4kHz = 0;
PV_total_4kHz = 0;

for i = 1:height(results);
    if string(results{i,2}) == 'Pyr'
        if string(results{i,5}) == '2' || string(results{i,6}) == '2'
            Pyr_total_2kHz = Pyr_total_2kHz + 1;
        end
        if string(results{i,5}) == '4' || string(results{i,6}) == '4'
            Pyr_total_4kHz = Pyr_total_4kHz + 1;
        end
        if string(results{i,47}) == 'yes'
            Pyr_yesTI_2kHz = Pyr_yesTI_2kHz + 1;
        end
        if string(results{i,48}) == 'yes' && string(results{i,8}) == 'NaN'
            Pyr_yesTI_4kHz = Pyr_yesTI_4kHz + 1;
        end
        Pyr_noTI_2kHz =  Pyr_total_2kHz - Pyr_yesTI_2kHz;
        Pyr_noTI_4kHz = Pyr_total_4kHz - Pyr_yesTI_4kHz;
    end
    if string(results{i,2}) == 'PV'
        if string(results{i,5}) == '2' || string(results{i,6}) == '2'
            PV_total_2kHz = PV_total_2kHz + 1;
        end
        if string(results{i,5}) == '4' || string(results{i,6}) == '4'
            PV_total_4kHz = PV_total_4kHz + 1;
        end
        if string(results{i,47}) == 'yes'
            PV_yesTI_2kHz = PV_yesTI_2kHz + 1;
        end
        if string(results{i,48}) == 'yes' && string(results{i,8}) == 'NaN'
            PV_yesTI_4kHz = PV_yesTI_4kHz + 1;
        end
        PV_noTI_2kHz = PV_total_2kHz - PV_yesTI_2kHz;
        PV_noTI_4kHz = PV_total_4kHz - PV_yesTI_4kHz;
    end

end

% 2 - Create pie charts
fig4a = figure;
PYR2kHz = [Pyr_yesTI_2kHz Pyr_noTI_2kHz];
PYR4kHz = [Pyr_yesTI_4kHz Pyr_noTI_4kHz];
PV2kHz = [PV_yesTI_2kHz PV_noTI_2kHz];
PV4kHz = [PV_yesTI_4kHz PV_noTI_4kHz];
labels = {'Exhibit TI',' Do not exhibit TI'};

subplot(2,2,1);
h = pie(PYR2kHz);
set(findobj(gca,'type','line'),'linew',2);
% title({'Pyr 2kHz',''});
% set(gca,'FontSize',20);
set(findobj(h,'type','text'),'fontsize',20);

subplot(2,2,2);
h = pie(PYR4kHz);
set(findobj(gca,'type','line'),'linew',2);
% title({'Pyr 4kHz',''}); %,'Color',[0.8500, 0.3250, 0.0980]);
% set(gca,'FontSize',20);
set(findobj(h,'type','text'),'fontsize',20);

subplot(2,2,3);
h = pie(PV2kHz);
set(findobj(gca,'type','line'),'linew',2);
% title({'PV 2kHz',''}); %,'Color',[0.8500, 0.3250, 0.0980]);
% set(gca,'FontSize',20);
set(findobj(h,'type','text'),'fontsize',20);

subplot(2,2,4);
h = pie(PV4kHz);
set(findobj(gca,'type','line'),'linew',2);
% title({'PV4 kHz',''}); %,'Color',[0.8500, 0.3250, 0.0980]);
% set(gca,'FontSize',20);
set(findobj(h,'type','text'),'fontsize',20);

colormap([1 1 1;0 0 0]); %([1 0.83 0; 0.35 0.26 0.49]);
hL = legend(labels,'Box','off','FontSize',16);
% newPosition = [0.5 0.15 0.1 0.1];
% newUnits = 'normalized';
set(hL,'Position', newPosition,'Units', newUnits);
saveas(gcf,'Fig3_eachFreqsPies.png');


%% Each boxplot
mainFolder = cd('C:\Users\chait\OneDrive\Desktop\SARA\specialCase\recheckingAll_240720\inputFiles'); % CHANGE directory accordingly where the inputFiles folder is
results = readtable('allCellsResultsSummary.xlsx');
cd(mainFolder); %% return to main directory that should be where the outputFiles folder is


% for 2kHz
Pyr_2kHz_normI_sin = [];
Pyr_2kHz_normI_mod = [];
PV_2kHz_normI_sin = [];
PV_2kHz_normI_mod = [];

Pyr_2kHz_FR_sin = [];
Pyr_2kHz_FR_mod = [];
PV_2kHz_FR_sin = [];
PV_2kHz_FR_mod = [];

Pyr_2kHz_ITI_sin = [];
Pyr_2kHz_ITI_mod = [];
PV_2kHz_ITI_sin = [];
PV_2kHz_ITI_mod = [];

for i = 1:height(results)

    % IF Pyr...
    if string(results{i,2}) == 'Pyr' && (string(results{i,6}) == '2' || string(results{i,5}) == '2')
        if string(results{i,47}) == 'yes'
            Pyr_2kHz_normI_mod = vertcat(Pyr_2kHz_normI_mod,results{i,28});
            Pyr_2kHz_FR_mod = vertcat(Pyr_2kHz_FR_mod,results{i,18});
            Pyr_2kHz_ITI_mod = vertcat(Pyr_2kHz_ITI_mod,results{i,17});
            if string(results{i,49}) == 'sin'
                Pyr_2kHz_normI_sin = vertcat(Pyr_2kHz_normI_sin,str2double(results{i,59}));
                Pyr_2kHz_FR_sin = vertcat(Pyr_2kHz_FR_sin,str2double(results{i,57}));
                Pyr_2kHz_ITI_sin = vertcat(Pyr_2kHz_ITI_sin,str2double(results{i,56}));
            end
        elseif string(results{i,47}) == 'no'
            Pyr_2kHz_normI_sin = vertcat(Pyr_2kHz_normI_sin,results{i,28});
            Pyr_2kHz_FR_sin = vertcat(Pyr_2kHz_FR_sin,results{i,26});
            Pyr_2kHz_ITI_sin = vertcat(Pyr_2kHz_ITI_sin,results{i,25});
            if string(results{i,49}) == 'mod'
                Pyr_2kHz_normI_mod = vertcat(Pyr_2kHz_normI_mod,str2double(results{i,59}));
                Pyr_2kHz_FR_mod = vertcat(Pyr_2kHz_FR_mod,str2double(results{i,57}));
                Pyr_2kHz_ITI_mod = vertcat(Pyr_2kHz_ITI_mod,str2double(results{i,56}));
            elseif string(results{i,49}) == 'same'
                Pyr_2kHz_normI_mod = vertcat(Pyr_2kHz_normI_mod,results{i,28});
                Pyr_2kHz_FR_mod = vertcat(Pyr_2kHz_FR_mod,results{i,18});
                Pyr_2kHz_ITI_mod = vertcat(Pyr_2kHz_ITI_mod,results{i,17});
            end
        end
    end

    % IF PV...
    if string(results{i,2}) == 'PV' && (string(results{i,6}) == '2' || string(results{i,5}) == '2')
        if string(results{i,47}) == 'yes'
            PV_2kHz_normI_mod = vertcat(PV_2kHz_normI_mod,results{i,28});
            PV_2kHz_FR_mod = vertcat(PV_2kHz_FR_mod,results{i,18});
            PV_2kHz_ITI_mod = vertcat(PV_2kHz_ITI_mod,results{i,17});
            if string(results{i,49}) == 'sin'
                PV_2kHz_normI_sin = vertcat(PV_2kHz_normI_sin,str2double(results{i,59}));
                PV_2kHz_FR_sin = vertcat(PV_2kHz_FR_sin,str2double(results{i,57}));
                PV_2kHz_ITI_sin = vertcat(PV_2kHz_ITI_sin,str2double(results{i,56}));
            end
        elseif string(results{i,47}) == 'no'
            PV_2kHz_normI_sin = vertcat(PV_2kHz_normI_sin,results{i,28});
            PV_2kHz_FR_sin = vertcat(PV_2kHz_FR_sin,results{i,26});
            PV_2kHz_ITI_sin = vertcat(PV_2kHz_ITI_sin,results{i,25});
            if string(results{i,49}) == 'mod'
                PV_2kHz_normI_mod = vertcat(PV_2kHz_normI_mod,str2double(results{i,59}));
                PV_2kHz_FR_mod = vertcat(PV_2kHz_FR_mod,str2double(results{i,57}));
                PV_2kHz_ITI_mod = vertcat(PV_2kHz_ITI_mod,str2double(results{i,56}));
            elseif string(results{i,49}) == 'same'
                PV_2kHz_normI_mod = vertcat(PV_2kHz_normI_mod,results{i,28});
                PV_2kHz_FR_mod = vertcat(PV_2kHz_FR_mod,results{i,18});
                PV_2kHz_ITI_mod = vertcat(PV_2kHz_ITI_mod,results{i,17});
            end
        end
    end
end

% means and stds that will be used in R script for statistics

Pyr_2kHz_normI_sin_mean = mean(Pyr_2kHz_normI_sin,'omitnan')
Pyr_2kHz_normI_mod_mean = mean(Pyr_2kHz_normI_mod,'omitnan')
PV_2kHz_normI_sin_mean = mean(PV_2kHz_normI_sin,'omitnan')
PV_2kHz_normI_mod_mean = mean(PV_2kHz_normI_mod,'omitnan')
Pyr_2kHz_normI_sin_std = std(Pyr_2kHz_normI_sin,'omitnan')
Pyr_2kHz_normI_mod_std = std(Pyr_2kHz_normI_mod,'omitnan')
PV_2kHz_normI_sin_std = std(PV_2kHz_normI_sin,'omitnan')
PV_2kHz_normI_mod_std = std(PV_2kHz_normI_mod,'omitnan')

Pyr_2kHz_FR_sin_mean = mean(Pyr_2kHz_FR_sin,'omitnan')
Pyr_2kHz_FR_mod_mean = mean(Pyr_2kHz_FR_mod,'omitnan')
PV_2kHz_FR_sin_mean = mean(PV_2kHz_FR_sin,'omitnan')
PV_2kHz_FR_mod_mean = mean(PV_2kHz_FR_mod,'omitnan')
Pyr_2kHz_FR_sin_std = std(Pyr_2kHz_FR_sin,'omitnan')
Pyr_2kHz_FR_mod_std = std(Pyr_2kHz_FR_mod,'omitnan')
PV_2kHz_FR_sin_std = std(PV_2kHz_FR_sin,'omitnan')
PV_2kHz_FR_mod_std = std(PV_2kHz_FR_mod,'omitnan')

Pyr_2kHz_ITI_sin_mean = mean(Pyr_2kHz_ITI_sin,'omitnan')
Pyr_2kHz_ITI_mod_mean = mean(Pyr_2kHz_ITI_mod,'omitnan')
PV_2kHz_ITI_sin_mean = mean(PV_2kHz_ITI_sin,'omitnan')
PV_2kHz_ITI_mod_mean = mean(PV_2kHz_ITI_mod,'omitnan')
Pyr_2kHz_ITI_sin_std = std(Pyr_2kHz_ITI_sin,'omitnan')
Pyr_2kHz_ITI_mod_std = std(Pyr_2kHz_ITI_mod,'omitnan')
PV_2kHz_ITI_sin_std = std(PV_2kHz_ITI_sin,'omitnan')
PV_2kHz_ITI_mod_std = std(PV_2kHz_ITI_mod,'omitnan')


% plots
for i = 1:length(Pyr_2kHz_FR_sin)
    Pyr_2kHz_sin(1,i) = 1;
end
for i = 1:length(Pyr_2kHz_FR_mod)
    Pyr_2kHz_mod(1,i) = 1;
end
for i = 1:length(Pyr_2kHz_normI_mod)
    Pyr_2kHz_mod_s(1,i) = 1;
end
for i = 1:length(PV_2kHz_FR_sin)
    PV_2kHz_sin(1,i) = 1;
end
for i = 1:length(PV_2kHz_FR_mod)
    PV_2kHz_mod(1,i) = 1;
end
for i = 1:length(PV_2kHz_normI_mod)
    PV_2kHz_mod_s(1,i) = 1;
end

fig3a = figure;
% boxes = boxplot(Pyr_2kHz_normI_sin, Pyr_2kHz_sin);
boxes = boxchart(Pyr_2kHz_normI_sin);
set(boxes, 'BoxFaceColor', 'none', 'BoxEdgeColor', 'k', 'MarkerStyle','+','MarkerColor','r'); hold on;
ylabel({'Pyr sin', 'Normalized Amplitude by Distance'}, 'FontSize', 18);
ylim([0 270]);
set(gca,'FontSize',18, 'LineWidth',2);
for n=1:max(unique( Pyr_2kHz_sin))
    hs = scatter(ones(sum( Pyr_2kHz_sin==n),1) + n-1, Pyr_2kHz_normI_sin( Pyr_2kHz_sin == n),"filled",'jitter','on','JitterAmount',0.1);
    hs.MarkerFaceAlpha = 0.5;
    hs.MarkerFaceColor = 'b'
end
box off
saveas(gcf,'Fig3_Pyr_2kHz_normI_sin.tif')
saveas(gcf,'Fig3_Pyr_2kHz_normI_sin.fig')

fig3a = figure;
% boxes = boxplot(Pyr_2kHz_normI_mod, Pyr_2kHz_mod_s);
boxes = boxchart(Pyr_2kHz_normI_mod);
set(boxes, 'BoxFaceColor', 'none', 'BoxEdgeColor', 'k', 'MarkerStyle','+','MarkerColor','r'); hold on;
ylabel({'Pyr mod', 'Normalized Amplitude by Distance'}, 'FontSize', 18);
ylim([0 270]);
set(gca,'FontSize',18, 'LineWidth',2);
for n=1:max(unique(Pyr_2kHz_mod))
    hs = scatter(ones(sum(Pyr_2kHz_mod==n),1) + n-1, Pyr_2kHz_normI_mod(Pyr_2kHz_mod == n),"filled",'jitter','on','JitterAmount',0.1);
    hs.MarkerFaceAlpha = 0.5;
    hs.MarkerFaceColor = 'b'
end
box off
saveas(gcf,'Fig3_Pyr_2kHz_normI_mod.tif')
saveas(gcf,'Fig3_Pyr_2kHz_normI_mod.fig')

fig3a = figure;
% boxes = boxplot(PV_2kHz_normI_sin, PV_2kHz_sin);
boxes = boxchart(PV_2kHz_normI_sin);
set(boxes, 'BoxFaceColor', 'none', 'BoxEdgeColor', 'k', 'MarkerStyle','+','MarkerColor','r'); hold on;
ylabel({'PV sin', 'Normalized Amplitude by Distance'}, 'FontSize', 18);
ylim([0 270]);
set(gca,'FontSize',18, 'LineWidth',2);
for n=1:max(unique(PV_2kHz_sin))
    hs = scatter(ones(sum(PV_2kHz_sin==n),1) + n-1, PV_2kHz_normI_sin(PV_2kHz_sin == n),"filled",'jitter','on','JitterAmount',0.1);
    hs.MarkerFaceAlpha = 0.5;
    hs.MarkerFaceColor = 'm'
end
box off
saveas(gcf,'Fig3_PV_2kHz_normI_sin.tif')
saveas(gcf,'Fig3_PV_2kHz_normI_sin.fig')

fig3a = figure;
% boxes = boxplot(PV_2kHz_normI_mod, PV_2kHz_mod_s);
boxes = boxchart(PV_2kHz_normI_mod);
set(boxes, 'BoxFaceColor', 'none', 'BoxEdgeColor', 'k', 'MarkerStyle','+','MarkerColor','r'); hold on;
ylabel({'PV mod', 'Normalized Amplitude by Distance'}, 'FontSize', 18);
ylim([0 270]);
set(gca,'FontSize',18, 'LineWidth',2);
for n=1:max(unique(PV_2kHz_mod))
    hs = scatter(ones(sum(PV_2kHz_mod==n),1) + n-1, PV_2kHz_normI_mod(PV_2kHz_mod == n),"filled",'jitter','on','JitterAmount',0.1);
    hs.MarkerFaceAlpha = 0.5;
    hs.MarkerFaceColor = 'm'
end
box off
saveas(gcf,'Fig3_PV_2kHz_normI_mod.tif')
saveas(gcf,'Fig3_PV_2kHz_normI_mod.fig')

fig3b = figure;
% boxes = boxplot(Pyr_2kHz_FR_sin, Pyr_2kHz_sin);
boxes = boxchart(Pyr_2kHz_FR_sin);
set(boxes, 'BoxFaceColor', 'none', 'BoxEdgeColor', 'k', 'MarkerStyle','+','MarkerColor','r'); hold on;
ylabel({'Pyr sin', 'Firing Rate (spks/s)'}, 'FontSize', 18);
ylim([0 110]);
set(gca,'FontSize',18, 'LineWidth',2);
for n=1:max(unique(Pyr_2kHz_sin))
    hs = scatter(ones(sum(Pyr_2kHz_sin==n),1) + n-1, Pyr_2kHz_FR_sin(Pyr_2kHz_sin == n),"filled",'jitter','on','JitterAmount',0.1);
    hs.MarkerFaceAlpha = 0.5;
    hs.MarkerFaceColor = 'b'
end
box off
saveas(gcf,'Fig3_Pyr_2kHz_FR_sin.tif')
saveas(gcf,'Fig3_Pyr_2kHz_FR_sin.fig')

fig3a = figure;
% boxes = boxplot(Pyr_2kHz_FR_mod, Pyr_2kHz_mod);
boxes = boxchart(Pyr_2kHz_FR_mod);
set(boxes, 'BoxFaceColor', 'none', 'BoxEdgeColor', 'k', 'MarkerStyle','+','MarkerColor','r'); hold on;
ylabel({'Pyr mod', 'Firing Rate (spks/s)'}, 'FontSize', 18);
ylim([0 110]);
set(gca,'FontSize',18, 'LineWidth',2);
for n=1:max(unique(Pyr_2kHz_mod))
    hs = scatter(ones(sum(Pyr_2kHz_mod==n),1) + n-1, Pyr_2kHz_FR_mod(Pyr_2kHz_mod == n),"filled",'jitter','on','JitterAmount',0.1);
    hs.MarkerFaceAlpha = 0.5;
    hs.MarkerFaceColor = 'b'
end
box off
saveas(gcf,'Fig3_Pyr_2kHz_FR_mod.tif')
saveas(gcf,'Fig3_Pyr_2kHz_FR_mod.fig')

fig3a = figure;
% boxes = boxplot(PV_2kHz_FR_sin, PV_2kHz_sin);
boxes = boxchart(PV_2kHz_FR_sin);
set(boxes, 'BoxFaceColor', 'none', 'BoxEdgeColor', 'k', 'MarkerStyle','+','MarkerColor','r'); hold on;
ylabel({'PV sin', 'Firing Rate (spks/s)'}, 'FontSize', 18);
ylim([0 110]);
set(gca,'FontSize',18, 'LineWidth',2);
for n=1:max(unique(PV_2kHz_sin))
    hs = scatter(ones(sum(PV_2kHz_sin==n),1) + n-1, PV_2kHz_FR_sin(PV_2kHz_sin == n),"filled",'jitter','on','JitterAmount',0.1);
    hs.MarkerFaceAlpha = 0.5;
    hs.MarkerFaceColor = 'm'
end
box off
saveas(gcf,'Fig3_PV_2kHz_FR_sin.tif')
saveas(gcf,'Fig3_PV_2kHz_FR_sin.fig')

fig3a = figure;
% boxes = boxplot(PV_2kHz_FR_mod, PV_2kHz_mod);
boxes = boxchart(PV_2kHz_FR_mod);
set(boxes, 'BoxFaceColor', 'none', 'BoxEdgeColor', 'k', 'MarkerStyle','+','MarkerColor','r'); hold on;
ylabel({'PV mod', 'Firing Rate (spks/s)'}, 'FontSize', 18);
ylim([0 110]);
set(gca,'FontSize',18, 'LineWidth',2);
for n=1:max(unique(PV_2kHz_mod))
    hs = scatter(ones(sum(PV_2kHz_mod==n),1) + n-1, PV_2kHz_FR_mod(PV_2kHz_mod == n),"filled",'jitter','on','JitterAmount',0.1);
    hs.MarkerFaceAlpha = 0.5;
    hs.MarkerFaceColor = 'm'
end
box off
saveas(gcf,'Fig3_PV_2kHz_FR_mod.tif')
saveas(gcf,'Fig3_PV_2kHz_FR_mod.fig')

fig3a = figure;
% boxes = boxplot(Pyr_2kHz_ITI_sin, Pyr_2kHz_sin);
boxes = boxchart(Pyr_2kHz_ITI_sin);
set(boxes, 'BoxFaceColor', 'none', 'BoxEdgeColor', 'k', 'MarkerStyle','+','MarkerColor','r'); hold on;
ylabel({'Pyr sin', 'Interspike Interval (s)'}, 'FontSize', 18);
ylim([0 0.12]);
set(gca,'FontSize',18, 'LineWidth',2);
for n=1:max(unique(Pyr_2kHz_sin))
    hs = scatter(ones(sum(Pyr_2kHz_sin==n),1) + n-1, Pyr_2kHz_ITI_sin(Pyr_2kHz_sin == n),"filled",'jitter','on','JitterAmount',0.1);
    hs.MarkerFaceAlpha = 0.5;
    hs.MarkerFaceColor = 'b'
end
box off
saveas(gcf,'Fig3_Pyr_2kHz_ITI_sin.tif')
saveas(gcf,'Fig3_Pyr_2kHz_ITI_sin.fig')

fig3a = figure;
% boxes = boxplot(Pyr_2kHz_ITI_mod, Pyr_2kHz_mod);
boxes = boxchart(Pyr_2kHz_ITI_mod);
set(boxes, 'BoxFaceColor', 'none', 'BoxEdgeColor', 'k', 'MarkerStyle','+','MarkerColor','r'); hold on;
ylabel({'Pyr mod', 'Interspike Interval (s)'}, 'FontSize', 18);
ylim([0 0.12]);
set(gca,'FontSize',18, 'LineWidth',2);
for n=1:max(unique(Pyr_2kHz_mod))
    hs = scatter(ones(sum(Pyr_2kHz_mod==n),1) + n-1, Pyr_2kHz_ITI_mod(Pyr_2kHz_mod == n),"filled",'jitter','on','JitterAmount',0.1);
    hs.MarkerFaceAlpha = 0.5;
    hs.MarkerFaceColor = 'b'
end
box off
saveas(gcf,'Fig3_Pyr_2kHz_ITI_mod.tif')
saveas(gcf,'Fig3_Pyr_2kHz_ITI_mod.fig')

fig3a = figure;
% boxes = boxplot(PV_2kHz_ITI_sin, PV_2kHz_sin);
boxes = boxchart(PV_2kHz_ITI_sin);
set(boxes, 'BoxFaceColor', 'none', 'BoxEdgeColor', 'k', 'MarkerStyle','+','MarkerColor','r'); hold on;
ylabel({'PV sin', 'Interspike Interval (s)'}, 'FontSize', 18);
ylim([0 0.12]);
set(gca,'FontSize',18, 'LineWidth',2);
for n=1:max(unique(PV_2kHz_sin))
    hs = scatter(ones(sum(PV_2kHz_sin==n),1) + n-1, PV_2kHz_ITI_sin(PV_2kHz_sin == n),"filled",'jitter','on','JitterAmount',0.1);
    hs.MarkerFaceAlpha = 0.5;
    hs.MarkerFaceColor = 'm'
end
box off
saveas(gcf,'Fig3_PV_2kHz_ITI_sin.tif')
saveas(gcf,'Fig3_PV_2kHz_ITI_sin.fig')

fig3a = figure;
% boxes = boxplot(PV_2kHz_ITI_mod, PV_2kHz_mod);
boxes = boxchart(PV_2kHz_ITI_mod);
set(boxes, 'BoxFaceColor', 'none', 'BoxEdgeColor', 'k', 'MarkerStyle','+','MarkerColor','r'); hold on;
ylabel({'PV mod','Interspike Interval (s)'}, 'FontSize', 18);
ylim([0 0.12]);
set(gca,'FontSize',18, 'LineWidth',2);
for n=1:max(unique(PV_2kHz_mod))
    hs = scatter(ones(sum(PV_2kHz_mod==n),1) + n-1, PV_2kHz_ITI_mod(PV_2kHz_mod == n),"filled",'jitter','on','JitterAmount',0.1);
    hs.MarkerFaceAlpha = 0.5;
    hs.MarkerFaceColor = 'm'
end
box off
saveas(gcf,'Fig3_PV_2kHz_ITI_mod.tif')
saveas(gcf,'Fig3_PV_2kHz_ITI_mod.fig')


% for 4kHz
Pyr_4kHz_normI_sin = [];
Pyr_4kHz_normI_mod = [];
PV_4kHz_normI_sin = [];
PV_4kHz_normI_mod = [];

Pyr_4kHz_FR_sin = [];
Pyr_4kHz_FR_mod = [];
PV_4kHz_FR_sin = [];
PV_4kHz_FR_mod = [];

Pyr_4kHz_ITI_sin = [];
Pyr_4kHz_ITI_mod = [];
PV_4kHz_ITI_sin = [];
PV_4kHz_ITI_mod = [];

for i = 1:height(results);

    % IF Pyr...
    if string(results{i,2}) == 'Pyr' && string(results{i,8}) == 'NaN' && (string(results{i,6}) == '4' || string(results{i,5}) == '4')
        if string(results{i,48}) == 'yes'
            Pyr_4kHz_normI_mod = vertcat(Pyr_4kHz_normI_mod,results{i,46});
            Pyr_4kHz_FR_mod = vertcat(Pyr_4kHz_FR_mod,results{i,36});
            Pyr_4kHz_ITI_mod = vertcat(Pyr_4kHz_ITI_mod,results{i,35});
            if string(results{i,60}) == 'sin'
                Pyr_4kHz_normI_sin = vertcat(Pyr_4kHz_normI_sin,str2double(results{i,70}));
                Pyr_4kHz_FR_sin = vertcat(Pyr_4kHz_FR_sin,str2double(results{i,68}));
                Pyr_4kHz_ITI_sin = vertcat(Pyr_4kHz_ITI_sin,str2double(results{i,67}));
            end
        elseif string(results{i,48}) == 'no'
            Pyr_4kHz_normI_sin = vertcat(Pyr_4kHz_normI_sin,results{i,46});
            Pyr_4kHz_FR_sin = vertcat(Pyr_4kHz_FR_sin,results{i,44});
            Pyr_4kHz_ITI_sin = vertcat(Pyr_4kHz_ITI_sin,results{i,43});
            if string(results{i,60}) == 'mod'
                Pyr_4kHz_normI_mod = vertcat(Pyr_4kHz_normI_mod,str2double(results{i,70}));
                Pyr_4kHz_FR_mod = vertcat(Pyr_4kHz_FR_mod,str2double(results{i,68}));
                Pyr_4kHz_ITI_mod = vertcat(Pyr_4kHz_ITI_mod,str2double(results{i,67}));
            elseif string(results{i,60}) == 'same'
                Pyr_4kHz_normI_mod = vertcat(Pyr_4kHz_normI_mod,results{i,46});
                Pyr_4kHz_FR_mod = vertcat(Pyr_4kHz_FR_mod,results{i,36});
                Pyr_4kHz_ITI_mod = vertcat(Pyr_4kHz_ITI_mod,results{i,35});
            end
        end
    end

    % IF PV...
    if string(results{i,2}) == 'PV' && string(results{i,8}) == 'NaN' && (string(results{i,6}) == '4' || string(results{i,5}) == '4')
        if string(results{i,48}) == 'yes'
            PV_4kHz_normI_mod = vertcat(PV_4kHz_normI_mod,results{i,46});
            PV_4kHz_FR_mod = vertcat(PV_4kHz_FR_mod,results{i,36});
            PV_4kHz_ITI_mod = vertcat(PV_4kHz_ITI_mod,results{i,35});
            if string(results{i,60}) == 'sin'
                PV_4kHz_normI_sin = vertcat(PV_4kHz_normI_sin,str2double(results{i,70}));
                PV_4kHz_FR_sin = vertcat(PV_4kHz_FR_sin,str2double(results{i,68}));
                PV_4kHz_ITI_sin = vertcat(PV_4kHz_ITI_sin,str2double(results{i,67}));
            end
        elseif string(results{i,48}) == 'no'
            PV_4kHz_normI_sin = vertcat(PV_4kHz_normI_sin,results{i,46});
            PV_4kHz_FR_sin = vertcat(PV_4kHz_FR_sin,results{i,44});
            PV_4kHz_ITI_sin = vertcat(PV_4kHz_ITI_sin,results{i,43});
            if string(results{i,60}) == 'mod'
                PV_4kHz_normI_mod = vertcat(PV_4kHz_normI_mod,str2double(results{i,70}));
                PV_4kHz_FR_mod = vertcat(PV_4kHz_FR_mod,str2double(results{i,68}));
                PV_4kHz_ITI_mod = vertcat(PV_4kHz_ITI_mod,str2double(results{i,67}));
            elseif string(results{i,60}) == 'same'
                PV_4kHz_normI_mod = vertcat(PV_4kHz_normI_mod,results{i,46});
                PV_4kHz_FR_mod = vertcat(PV_4kHz_FR_mod,results{i,36});
                PV_4kHz_ITI_mod = vertcat(PV_4kHz_ITI_mod,results{i,35});
            end
        end
    end
end

% means and stds that will be used in R script for statistics

Pyr_4kHz_normI_sin_mean = mean(Pyr_4kHz_normI_sin,'omitnan')
Pyr_4kHz_normI_mod_mean = mean(Pyr_4kHz_normI_mod,'omitnan')
PV_4kHz_normI_sin_mean = mean(PV_4kHz_normI_sin,'omitnan')
PV_4kHz_normI_mod_mean = mean(PV_4kHz_normI_mod,'omitnan')
Pyr_4kHz_normI_sin_std = std(Pyr_4kHz_normI_sin,'omitnan')
Pyr_4kHz_normI_mod_std = std(Pyr_4kHz_normI_mod,'omitnan')
PV_4kHz_normI_sin_std = std(PV_4kHz_normI_sin,'omitnan')
PV_4kHz_normI_mod_std = std(PV_4kHz_normI_mod,'omitnan')


Pyr_4kHz_FR_sin_mean = mean(Pyr_4kHz_FR_sin,'omitnan')
Pyr_4kHz_FR_mod_mean = mean(Pyr_4kHz_FR_mod,'omitnan')
PV_4kHz_FR_sin_mean = mean(PV_4kHz_FR_sin,'omitnan')
PV_4kHz_FR_mod_mean = mean(PV_4kHz_FR_mod,'omitnan')
Pyr_4kHz_FR_sin_std = std(Pyr_4kHz_FR_sin,'omitnan')
Pyr_4kHz_FR_mod_std = std(Pyr_4kHz_FR_mod,'omitnan')
PV_4kHz_FR_sin_std = std(PV_4kHz_FR_sin,'omitnan')
PV_4kHz_FR_mod_std = std(PV_4kHz_FR_mod,'omitnan')

Pyr_4kHz_ITI_sin_mean = mean(Pyr_4kHz_ITI_sin,'omitnan')
Pyr_4kHz_ITI_mod_mean = mean(Pyr_4kHz_ITI_mod,'omitnan')
PV_4kHz_ITI_sin_mean = mean(PV_4kHz_ITI_sin,'omitnan')
PV_4kHz_ITI_mod_mean = mean(PV_4kHz_ITI_mod,'omitnan')
Pyr_4kHz_ITI_sin_std = std(Pyr_4kHz_ITI_sin,'omitnan')
Pyr_4kHz_ITI_mod_std = std(Pyr_4kHz_ITI_mod,'omitnan')
PV_4kHz_ITI_sin_std = std(PV_4kHz_ITI_sin,'omitnan')
PV_4kHz_ITI_mod_std = std(PV_4kHz_ITI_mod,'omitnan')


% plots
for i = 1:length(Pyr_4kHz_FR_sin)
    Pyr_4kHz_sin(1,i) = 1;
end
for i = 1:length(Pyr_4kHz_FR_mod)
    Pyr_4kHz_mod(1,i) = 1;
end
for i = 1:length(Pyr_4kHz_normI_mod)
    Pyr_4kHz_mod_s(1,i) = 1;
end
for i = 1:length(PV_4kHz_FR_sin)
    PV_4kHz_sin(1,i) = 1;
end
for i = 1:length(PV_4kHz_FR_mod)
    PV_4kHz_mod(1,i) = 1;
end
for i = 1:length(PV_4kHz_normI_mod)
    PV_4kHz_mod_s(1,i) = 1;
end

fig1 = figure;
% boxes = boxplot(Pyr_4kHz_normI_sin, Pyr_4kHz_sin);
boxes = boxchart(Pyr_4kHz_normI_sin);
set(boxes, 'BoxFaceColor', 'none', 'BoxEdgeColor', 'k', 'MarkerStyle','+','MarkerColor','r'); hold on;
ylabel({'Pyr sin', 'Normalized Amplitude by Distance'}, 'FontSize', 18);
ylim([0 270]);
set(gca,'FontSize',18, 'LineWidth',2);
for n=1:max(unique(Pyr_4kHz_sin))
    hs = scatter(ones(sum( Pyr_4kHz_sin==n),1) + n-1, Pyr_4kHz_normI_sin( Pyr_4kHz_sin == n),"filled",'jitter','on','JitterAmount',0.1);
    hs.MarkerFaceAlpha = 0.5;
    hs.MarkerFaceColor = 'b'
end
box off
saveas(gcf,'Fig3_Pyr_4kHz_normI_sin.tif')
saveas(gcf,'Fig3_Pyr_4kHz_normI_sin.fig')

fig2 = figure;
% boxes = boxplot(Pyr_4kHz_normI_mod, Pyr_4kHz_mod_s);
boxes = boxchart(Pyr_4kHz_normI_mod);
set(boxes, 'BoxFaceColor', 'none', 'BoxEdgeColor', 'k', 'MarkerStyle','+','MarkerColor','r'); hold on;
ylabel({'Pyr mod', 'Normalized Amplitude by Distance'}, 'FontSize', 18);
ylim([0 270]);
set(gca,'FontSize',18, 'LineWidth',2);
for n=1:max(unique(Pyr_4kHz_mod))
    hs = scatter(ones(sum(Pyr_4kHz_mod==n),1) + n-1, Pyr_4kHz_normI_mod(Pyr_4kHz_mod == n),"filled",'jitter','on','JitterAmount',0.1);
    hs.MarkerFaceAlpha = 0.5;
    hs.MarkerFaceColor = 'b'
end
box off
saveas(gcf,'Fig3_Pyr_4kHz_normI_mod.tif')
saveas(gcf,'Fig3_Pyr_4kHz_normI_mod.fig')

fig3 = figure;
% boxes = boxplot(PV_4kHz_normI_sin, PV_4kHz_sin);
boxes = boxchart(PV_4kHz_normI_sin);
set(boxes, 'BoxFaceColor', 'none', 'BoxEdgeColor', 'k', 'MarkerStyle','+','MarkerColor','r'); hold on;
ylabel({'PV sin', 'Normalized Amplitude by Distance'}, 'FontSize', 18);
ylim([0 270]);
set(gca,'FontSize',18, 'LineWidth',2);
for n=1:max(unique(PV_4kHz_sin))
    hs = scatter(ones(sum(PV_4kHz_sin==n),1) + n-1, PV_4kHz_normI_sin(PV_4kHz_sin == n),"filled",'jitter','on','JitterAmount',0.1);
    hs.MarkerFaceAlpha = 0.5;
    hs.MarkerFaceColor = 'm'
end
box off
saveas(gcf,'Fig3_PV_4kHz_normI_sin.tif')
saveas(gcf,'Fig3_PV_4kHz_normI_sin.fig')

fig4 = figure;
% boxes = boxplot(PV_4kHz_normI_mod, PV_4kHz_mod_s);
boxes = boxchart(PV_4kHz_normI_mod);
set(boxes, 'BoxFaceColor', 'none', 'BoxEdgeColor', 'k', 'MarkerStyle','+','MarkerColor','r'); hold on;
ylabel({'PV mod', 'Normalized Amplitude by Distance'}, 'FontSize', 18);
ylim([0 270]);
set(gca,'FontSize',18, 'LineWidth',2);
for n=1:max(unique(PV_4kHz_mod))
    hs = scatter(ones(sum(PV_4kHz_mod==n),1) + n-1, PV_4kHz_normI_mod(PV_4kHz_mod == n),"filled",'jitter','on','JitterAmount',0.1);
    hs.MarkerFaceAlpha = 0.5;
    hs.MarkerFaceColor = 'm'
end
box off
saveas(gcf,'Fig3_PV_4kHz_normI_mod.tif')
saveas(gcf,'Fig3_PV_4kHz_normI_mod.fig')

fig5 = figure;
% boxes = boxplot(Pyr_4kHz_FR_sin, Pyr_4kHz_sin);
boxes = boxchart(Pyr_4kHz_FR_sin);
set(boxes, 'BoxFaceColor', 'none', 'BoxEdgeColor', 'k', 'MarkerStyle','+','MarkerColor','r'); hold on;
ylabel({'Pyr sin', 'Firing Rate (spks/s)'}, 'FontSize', 18);
ylim([0 110]);
set(gca,'FontSize',18, 'LineWidth',2);
for n=1:max(unique(Pyr_4kHz_sin))
    hs = scatter(ones(sum(Pyr_4kHz_sin==n),1) + n-1, Pyr_4kHz_FR_sin(Pyr_4kHz_sin == n),"filled",'jitter','on','JitterAmount',0.1);
    hs.MarkerFaceAlpha = 0.5;
    hs.MarkerFaceColor = 'b'
end
box off
saveas(gcf,'Fig3_Pyr_4kHz_FR_sin.tif')
saveas(gcf,'Fig3_Pyr_4kHz_FR_sin.fig')


fig6 = figure;
% boxes = boxplot(Pyr_4kHz_FR_mod, Pyr_4kHz_mod);
boxes = boxchart(Pyr_4kHz_FR_mod);
set(boxes, 'BoxFaceColor', 'none', 'BoxEdgeColor', 'k', 'MarkerStyle','+','MarkerColor','r'); hold on;
ylabel({'Pyr mod', 'Firing Rate (spks/s)'}, 'FontSize', 18);
ylim([0 110]);
set(gca,'FontSize',18, 'LineWidth',2);
for n=1:max(unique(Pyr_4kHz_mod))
    hs = scatter(ones(sum(Pyr_4kHz_mod==n),1) + n-1, Pyr_4kHz_FR_mod(Pyr_4kHz_mod == n),"filled",'jitter','on','JitterAmount',0.1);
    hs.MarkerFaceAlpha = 0.5;
    hs.MarkerFaceColor = 'b'
end
box off
saveas(gcf,'Fig3_Pyr_4kHz_FR_mod.tif')
saveas(gcf,'Fig3_Pyr_4kHz_FR_mod.fig')

fig7 = figure;
% boxes = boxplot(PV_4kHz_FR_sin, PV_4kHz_sin);
boxes = boxchart(PV_4kHz_FR_sin);
set(boxes, 'BoxFaceColor', 'none', 'BoxEdgeColor', 'k', 'MarkerStyle','+','MarkerColor','r'); hold on;
ylabel({'PV sin', 'Firing Rate (spks/s)'}, 'FontSize', 18);
ylim([0 110]);
set(gca,'FontSize',18, 'LineWidth',2);
for n=1:max(unique(PV_4kHz_sin))
    hs = scatter(ones(sum(PV_4kHz_sin==n),1) + n-1, PV_4kHz_FR_sin(PV_4kHz_sin == n),"filled",'jitter','on','JitterAmount',0.1);
    hs.MarkerFaceAlpha = 0.5;
    hs.MarkerFaceColor = 'm'
end
box off
saveas(gcf,'Fig3_PV_4kHz_FR_sin.tif')
saveas(gcf,'Fig3_PV_4kHz_FR_sin.fig')

fig8 = figure;
% boxes = boxplot(PV_4kHz_FR_mod, PV_4kHz_mod);
boxes = boxchart(PV_4kHz_FR_mod);
set(boxes, 'BoxFaceColor', 'none', 'BoxEdgeColor', 'k', 'MarkerStyle','+','MarkerColor','r'); hold on;
ylabel({'PV mod', 'Firing Rate (spks/s)'}, 'FontSize', 18);
ylim([0 110]);
set(gca,'FontSize',18, 'LineWidth',2);
for n=1:max(unique(PV_4kHz_mod))
    hs = scatter(ones(sum(PV_4kHz_mod==n),1) + n-1, PV_4kHz_FR_mod(PV_4kHz_mod == n),"filled",'jitter','on','JitterAmount',0.1);
    hs.MarkerFaceAlpha = 0.5;
    hs.MarkerFaceColor = 'm'
end
box off
saveas(gcf,'Fig3_PV_4kHz_FR_mod.tif')
saveas(gcf,'Fig3_PV_4kHz_FR_mod.fig')

fig9 = figure;
% boxes = boxplot(Pyr_4kHz_ITI_sin, Pyr_4kHz_sin);
boxes = boxchart(Pyr_4kHz_ITI_sin);
set(boxes, 'BoxFaceColor', 'none', 'BoxEdgeColor', 'k', 'MarkerStyle','+','MarkerColor','r'); hold on;
ylabel({'Pyr sin', 'Interspike Interval (s)'}, 'FontSize', 18);
ylim([0 0.12]);
set(gca,'FontSize',18, 'LineWidth',2);
for n=1:max(unique(Pyr_4kHz_sin))
    hs = scatter(ones(sum(Pyr_4kHz_sin==n),1) + n-1, Pyr_4kHz_ITI_sin(Pyr_4kHz_sin == n),"filled",'jitter','on','JitterAmount',0.1);
    hs.MarkerFaceAlpha = 0.5;
    hs.MarkerFaceColor = 'b'
end
box off
saveas(gcf,'Fig3_Pyr_4kHz_ITI_sin.tif')
saveas(gcf,'Fig3_Pyr_4kHz_ITI_sin.fig')

fig10 = figure;
% boxes = boxplot(Pyr_4kHz_ITI_mod, Pyr_4kHz_mod);
boxes = boxchart(Pyr_4kHz_ITI_mod);
set(boxes, 'BoxFaceColor', 'none', 'BoxEdgeColor', 'k', 'MarkerStyle','+','MarkerColor','r'); hold on;
ylabel({'Pyr mod', 'Interspike Interval (s)'}, 'FontSize', 18);
ylim([0 0.12]);
set(gca,'FontSize',18, 'LineWidth',2);
for n=1:max(unique(Pyr_4kHz_mod))
    hs = scatter(ones(sum(Pyr_4kHz_mod==n),1) + n-1, Pyr_4kHz_ITI_mod(Pyr_4kHz_mod == n),"filled",'jitter','on','JitterAmount',0.1);
    hs.MarkerFaceAlpha = 0.5;
    hs.MarkerFaceColor = 'b'
end
box off
saveas(gcf,'Fig3_Pyr_4kHz_ITI_mod.tif')
saveas(gcf,'Fig3_Pyr_4kHz_ITI_mod.fig')

fig11 = figure;
% boxes = boxplot(PV_4kHz_ITI_sin, PV_4kHz_sin);
boxes = boxchart(PV_4kHz_ITI_sin);
set(boxes, 'BoxFaceColor', 'none', 'BoxEdgeColor', 'k', 'MarkerStyle','+','MarkerColor','r'); hold on;
ylabel({'PV sin', 'Interspike Interval (s)'}, 'FontSize', 18);
ylim([0 0.12]);
set(gca,'FontSize',18, 'LineWidth',2);
for n=1:max(unique(PV_4kHz_sin))
    hs = scatter(ones(sum(PV_4kHz_sin==n),1) + n-1, PV_4kHz_ITI_sin(PV_4kHz_sin == n),"filled",'jitter','on','JitterAmount',0.1);
    hs.MarkerFaceAlpha = 0.5;
    hs.MarkerFaceColor = 'm'
end
box off
saveas(gcf,'Fig3_PV_4kHz_ITI_sin.tif')
saveas(gcf,'Fig3_PV_4kHz_ITI_sin.fig')

fig12 = figure;
% boxes = boxplot(PV_4kHz_ITI_mod, PV_4kHz_mod);
boxes = boxchart(PV_4kHz_ITI_mod);
set(boxes, 'BoxFaceColor', 'none', 'BoxEdgeColor', 'k', 'MarkerStyle','+','MarkerColor','r'); hold on;
ylabel({'PV mod','Interspike Interval (s)'}, 'FontSize', 18);
ylim([0 0.12]);
set(gca,'FontSize',18, 'LineWidth',2);
for n=1:max(unique(PV_4kHz_mod))
    hs = scatter(ones(sum(PV_4kHz_mod==n),1) + n-1, PV_4kHz_ITI_mod(PV_4kHz_mod == n),"filled",'jitter','on','JitterAmount',0.1);
    hs.MarkerFaceAlpha = 0.5;
    hs.MarkerFaceColor = 'm'
end
box off
saveas(gcf,'Fig3_PV_4kHz_ITI_mod.tif')
saveas(gcf,'Fig3_PV_4kHz_ITI_mod.fig')


%% Fig4 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Pyr exhibits TI --> does NOT exhibit TI
% 20210520b

% 1 - Extract data from txt file
mainFolder = cd('C:\Users\chait\OneDrive\Desktop\SARA\specialCase\recheckingAll_240720\inputFiles'); % CHANGE directory accordingly where the inputFiles folder is
% ephys file
part1Rec = readtable("SCM_20210520b_2abCL2-3_seIr-Pt4f_{0}.txt");
part2Rec = readtable("SCM_20210520b_2abCL2-3_seIr-Pt4f_{1}.txt");
% notes file
notes = readtable('20210520b_Notes.txt');
selectedIntraSweep = 16
selectedSinSweep = (table2array(notes(17,1)))+1
selectedTISweep = (table2array(notes(9,1)))+2
selectedSinSweep_afterDrug = (table2array(notes(37,1)))
selectedTISweep_afterDrug = (table2array(notes(29,1)))
cd(mainFolder); %% return to main directory

% 2 - Prepare data
newnames = matlab.lang.makeUniqueStrings([part1Rec.Properties.VariableNames, part2Rec.Properties.VariableNames]);
part1Rec.Properties.VariableNames = newnames(1:numel(part1Rec.Properties.VariableNames));
part2Rec.Properties.VariableNames = newnames(numel(part1Rec.Properties.VariableNames)+1:end);
Rec = [part1Rec part2Rec];
% remove NaN columns
badColumns = isnan(Rec{1,:});
CleanRec = Rec(:,~badColumns);
% segregate Time and Voltage sweeps
Time_Sweeps = CleanRec(:,1:2:end-1);
Voltage_Sweeps = CleanRec(:,2:2:end);

% 3 - Plot traces
% firing pattern --> intracellular stimulation
currentVoltage_Sweep = table2array(Voltage_Sweeps(:,selectedIntraSweep));
currentTime_Sweep = table2array(Time_Sweeps(:,selectedIntraSweep));
[b,a] = butter(3,[100,1000]/(10000/2));
filtVoltage_Sweep = filtfilt(b,a,currentVoltage_Sweep);
[pks,locs] = findpeaks(filtVoltage_Sweep, currentTime_Sweep, 'MinPeakProminence',20, "MinPeakHeight",15);
NumPeaks_Sweeps = length(pks);

Fig1c1 = figure;
x = currentTime_Sweep;
y1 = currentVoltage_Sweep;
plot(x,y1,'black','LineWidth',1);
title({'PYR', 'Intra Stim','exhibits TI'});
xlim([[1.8 2.7]]);
ylim([-130 100]);
xlabel('Time (s)', 'FontSize', 14);
ylabel('Voltage (mV)', 'FontSize', 14);
set(gca,'FontSize',14);
box off;
axis off;
saveas(gcf,'Fig4_20210520b_PYR_IntraStim_exhibitsTI.png');

currentVoltage_Sweep = table2array(Voltage_Sweeps(:,selectedSinSweep));
currentTime_Sweep = table2array(Time_Sweeps(:,selectedSinSweep));
[b,a] = butter(3,[100,1000]/(10000/2));
filtVoltage_Sweep = filtfilt(b,a,currentVoltage_Sweep);
[pks,locs] = findpeaks(filtVoltage_Sweep, currentTime_Sweep, 'MinPeakProminence',20, "MinPeakHeight",15);
NumPeaks_Sweeps = length(pks);

Fig1d1 = figure;
x = currentTime_Sweep;
y1 = currentVoltage_Sweep;
plot(x,y1,'black','LineWidth',1);
title({'PYR', 'Sin Stim','exhibits TI'});
x1 = (table2array(notes(20,1))) - 0.2
x2 = x1 + 0.9
xlim([x1 x2]);
ylim([-130 100]);
xlabel('Time (s)', 'FontSize', 14);
ylabel('Voltage (mV)', 'FontSize', 14);
set(gca,'FontSize',14);
box off;
axis off;
saveas(gcf,'Fig4_20210520b_PYR_SinStim_exhibitsTI.png');

currentVoltage_Sweep = table2array(Voltage_Sweeps(:,selectedTISweep));
currentTime_Sweep = table2array(Time_Sweeps(:,selectedTISweep));
[b,a] = butter(3,[100,1000]/(10000/2));
filtVoltage_Sweep = filtfilt(b,a,currentVoltage_Sweep);
[pks,locs] = findpeaks(filtVoltage_Sweep, currentTime_Sweep, 'MinPeakProminence',20, "MinPeakHeight",15);
NumPeaks_Sweeps = length(pks);

Fig1e1 = figure;
x = currentTime_Sweep;
y1 = currentVoltage_Sweep;
plot(x,y1,'black','LineWidth',1);
title({'PYR', 'TI Stim','exhibits TI'});
x1 = (table2array(notes(13,1))) - 0.2
x2 = x1 + 0.9
xlim([x1 x2]);
ylim([-130 100]);
xlabel('Time (s)', 'FontSize', 14);
ylabel('Voltage (mV)', 'FontSize', 14);
set(gca,'FontSize',14);
box off;
axis off;
saveas(gcf,'Fig4_20210520b_PYR_TIStim_exhibitsTI.png');

currentVoltage_Sweep = table2array(Voltage_Sweeps(:,selectedTISweep_afterDrug));
currentTime_Sweep = table2array(Time_Sweeps(:,selectedTISweep_afterDrug));
[b,a] = butter(3,[100,1000]/(10000/2));
filtVoltage_Sweep = filtfilt(b,a,currentVoltage_Sweep);
[pks,locs] = findpeaks(filtVoltage_Sweep, currentTime_Sweep, 'MinPeakProminence',20, "MinPeakHeight",15);
NumPeaks_Sweeps = length(pks);

Fig1f = figure;
x = currentTime_Sweep;
y1 = currentVoltage_Sweep;
plot(x,y1,'black','LineWidth',1);
title({'PYR', 'TI Stim','exhibits TI'});
x1 = (table2array(notes(39,1))) - 0.2
x2 = x1 + 0.9
xlim([x1 x2]);
ylim([-130 100]);
xlabel('Time (s)', 'FontSize', 14);
ylabel('Voltage (mV)', 'FontSize', 14);
set(gca,'FontSize',14);
box off;
axis off;
saveas(gcf,'Fig4_20210520b_PYR_TIStim_afterDrug.png');

currentVoltage_Sweep = table2array(Voltage_Sweeps(:,selectedSinSweep_afterDrug));
currentTime_Sweep = table2array(Time_Sweeps(:,selectedSinSweep_afterDrug));
[b,a] = butter(3,[100,1000]/(10000/2));
filtVoltage_Sweep = filtfilt(b,a,currentVoltage_Sweep);
[pks,locs] = findpeaks(filtVoltage_Sweep, currentTime_Sweep, 'MinPeakProminence',20, "MinPeakHeight",15);
NumPeaks_Sweeps = length(pks);

Fig1d1 = figure;
x = currentTime_Sweep;
y1 = currentVoltage_Sweep;
plot(x,y1,'black','LineWidth',1);
title({'PYR', 'Sin Stim','exhibits TI'});
x1 = (table2array(notes(31,1))) - 0.2
x2 = x1 + 0.9
xlim([x1 x2]);
ylim([-130 100]);
xlabel('Time (s)', 'FontSize', 14);
ylabel('Voltage (mV)', 'FontSize', 14);
set(gca,'FontSize',14);
box off;
axis off;
saveas(gcf,'Fig4_20210520b_PYR_SinStim_afterDrug.png');


%% PV does not exhibits TI
% 20210807a

% 1 - Extract data from txt file
mainFolder = cd('C:\Users\chait\OneDrive\Desktop\SARA\specialCase\recheckingAll_240720\inputFiles'); % CHANGE directory accordingly where the inputFiles folder is
% ephys file
part1Rec = readtable("SCM_20210807a_2aaCL2-3_seIr-Pt2f_{0}.txt"); %%15,15
part2Rec = readtable("SCM_20210807a_2aaCL2-3_seIr-Pt2f_{1}.txt");
% notes file
notes = readtable('20210807a_Notes.txt');
selectedIntraSweep = 16
selectedSinSweep = (table2array(notes(17,1)))+1
selectedTISweep = (table2array(notes(9,1)))+2
selectedSinSweep_afterDrug = (table2array(notes(37,1)))
selectedTISweep_afterDrug = (table2array(notes(29,1)))
cd(mainFolder); %% return to main directory

% 2 - Prepare data
newnames = matlab.lang.makeUniqueStrings([part1Rec.Properties.VariableNames, part2Rec.Properties.VariableNames]);
part1Rec.Properties.VariableNames = newnames(1:numel(part1Rec.Properties.VariableNames));
part2Rec.Properties.VariableNames = newnames(numel(part1Rec.Properties.VariableNames)+1:end);
Rec = [part1Rec part2Rec];
% remove NaN columns
badColumns = isnan(Rec{1,:});
CleanRec = Rec(:,~badColumns);
% segregate Time and Voltage sweeps
Time_Sweeps = CleanRec(:,1:2:end-1);
Voltage_Sweeps = CleanRec(:,2:2:end);

% 3 - Plot traces
% firing pattern --> intracellular stimulation
currentVoltage_Sweep = table2array(Voltage_Sweeps(:,selectedIntraSweep));
currentTime_Sweep = table2array(Time_Sweeps(:,selectedIntraSweep));
[b,a] = butter(3,[100,1000]/(10000/2));
filtVoltage_Sweep = filtfilt(b,a,currentVoltage_Sweep);
[pks,locs] = findpeaks(filtVoltage_Sweep, currentTime_Sweep, 'MinPeakProminence',20, "MinPeakHeight",15);
NumPeaks_Sweeps = length(pks);

Fig1c1 = figure;
x = currentTime_Sweep;
y1 = currentVoltage_Sweep;
plot(x,y1,'black','LineWidth',1);
title({'PYR', 'Intra Stim','exhibits TI'});
xlim([[1.8 2.7]]);
ylim([-130 100]);
xlabel('Time (s)', 'FontSize', 14);
ylabel('Voltage (mV)', 'FontSize', 14);
set(gca,'FontSize',14);
box off;
axis off;
saveas(gcf,'Fig4_20210807a_PV_IntraStim_exhibitsTI.png');

currentVoltage_Sweep = table2array(Voltage_Sweeps(:,selectedSinSweep));
currentTime_Sweep = table2array(Time_Sweeps(:,selectedSinSweep));
[b,a] = butter(3,[100,1000]/(10000/2));
filtVoltage_Sweep = filtfilt(b,a,currentVoltage_Sweep);
[pks,locs] = findpeaks(filtVoltage_Sweep, currentTime_Sweep, 'MinPeakProminence',20, "MinPeakHeight",15);
NumPeaks_Sweeps = length(pks);

Fig1d1 = figure;
x = currentTime_Sweep;
y1 = currentVoltage_Sweep;
plot(x,y1,'black','LineWidth',1);
title({'PYR', 'Sin Stim','exhibits TI'});
x1 = (table2array(notes(20,1))) - 0.2
x2 = x1 + 0.9
xlim([x1 x2]);
ylim([-130 100]);
xlabel('Time (s)', 'FontSize', 14);
ylabel('Voltage (mV)', 'FontSize', 14);
set(gca,'FontSize',14);
box off;
axis off;
saveas(gcf,'Fig4_20210807a_PV_SinStim_exhibitsTI.png');

currentVoltage_Sweep = table2array(Voltage_Sweeps(:,selectedTISweep));
currentTime_Sweep = table2array(Time_Sweeps(:,selectedTISweep));
[b,a] = butter(3,[100,1000]/(10000/2));
filtVoltage_Sweep = filtfilt(b,a,currentVoltage_Sweep);
[pks,locs] = findpeaks(filtVoltage_Sweep, currentTime_Sweep, 'MinPeakProminence',20, "MinPeakHeight",15);
NumPeaks_Sweeps = length(pks);

Fig1e1 = figure;
x = currentTime_Sweep;
y1 = currentVoltage_Sweep;
plot(x,y1,'black','LineWidth',1);
title({'PYR', 'TI Stim','exhibits TI'});
x1 = (table2array(notes(13,1))) - 0.2
x2 = x1 + 0.9
xlim([x1 x2]);
ylim([-130 100]);
xlabel('Time (s)', 'FontSize', 14);
ylabel('Voltage (mV)', 'FontSize', 14);
set(gca,'FontSize',14);
box off;
axis off;
saveas(gcf,'Fig4_20210807a_PV_TIStim_exhibitsTI.png');

currentVoltage_Sweep = table2array(Voltage_Sweeps(:,selectedTISweep_afterDrug));
currentTime_Sweep = table2array(Time_Sweeps(:,selectedTISweep_afterDrug));
[b,a] = butter(3,[100,1000]/(10000/2));
filtVoltage_Sweep = filtfilt(b,a,currentVoltage_Sweep);
[pks,locs] = findpeaks(filtVoltage_Sweep, currentTime_Sweep, 'MinPeakProminence',20, "MinPeakHeight",15);
NumPeaks_Sweeps = length(pks);

Fig1f = figure;
x = currentTime_Sweep;
y1 = currentVoltage_Sweep;
plot(x,y1,'black','LineWidth',1);
title({'PYR', 'TI Stim','exhibits TI'});
x1 = (table2array(notes(39,1))) - 0.2
x2 = x1 + 0.9
xlim([x1 x2]);
ylim([-130 100]);
xlabel('Time (s)', 'FontSize', 14);
ylabel('Voltage (mV)', 'FontSize', 14);
set(gca,'FontSize',14);
box off;
axis off;
saveas(gcf,'Fig4_20210807a_PV_TIStim_afterDrug.png');

currentVoltage_Sweep = table2array(Voltage_Sweeps(:,selectedSinSweep_afterDrug));
currentTime_Sweep = table2array(Time_Sweeps(:,selectedSinSweep_afterDrug));
[b,a] = butter(3,[100,1000]/(10000/2));
filtVoltage_Sweep = filtfilt(b,a,currentVoltage_Sweep);
[pks,locs] = findpeaks(filtVoltage_Sweep, currentTime_Sweep, 'MinPeakProminence',20, "MinPeakHeight",15);
NumPeaks_Sweeps = length(pks);

Fig1d1 = figure;
x = currentTime_Sweep;
y1 = currentVoltage_Sweep;
plot(x,y1,'black','LineWidth',1);
title({'PYR', 'Sin Stim','exhibits TI'});
x1 = (table2array(notes(31,1))) - 0.2
x2 = x1 + 0.9
xlim([x1 x2]);
ylim([-130 100]);
xlabel('Time (s)', 'FontSize', 14);
ylabel('Voltage (mV)', 'FontSize', 14);
set(gca,'FontSize',14);
box off;
axis off;
saveas(gcf,'Fig4_20210807a_PV_SinStim_afterDrug.png');


%% Does or does not exhibit TI PIE CHARTS _beforeDrugs
mainFolder = cd('C:\Users\chait\OneDrive\Desktop\SARA\specialCase\recheckingAll_240720\inputFiles'); % CHANGE directory accordingly where the inputFiles folder is
results = readtable('allCellsResultsSummary.xlsx');
cd(mainFolder); %% return to main directory that should be where the outputFiles folder is

Pyr_yesTI_beforeDrugs = 0;
Pyr_noTI_beforeDrugs = 0;
Pyr_total_Drugs = 0;

PV_yesTI_beforeDrugs = 0;
PV_noTI_beforeDrugs = 0;
PV_total_Drugs = 0;

for i = 1:height(results);
    if string(results{i,2}) == 'Pyr'
        if string(results{i,8}) == 'beforeDrugs'
            Pyr_total_Drugs = Pyr_total_Drugs + 1;
            if string(results{i,47}) == 'yes'
                Pyr_yesTI_beforeDrugs = Pyr_yesTI_beforeDrugs + 1;
            end
            Pyr_noTI_beforeDrugs = Pyr_total_Drugs - Pyr_yesTI_beforeDrugs;
        end
    end
    if string(results{i,2}) == 'PV'
        if string(results{i,8}) == 'beforeDrugs'
            PV_total_Drugs = PV_total_Drugs + 1;
            if string(results{i,47}) == 'yes'
                PV_yesTI_beforeDrugs = PV_yesTI_beforeDrugs + 1;
            end
            PV_noTI_beforeDrugs = PV_total_Drugs - PV_yesTI_beforeDrugs;
        end
    end
end

Fig2x1 = figure;
Pyr = [Pyr_yesTI_beforeDrugs Pyr_noTI_beforeDrugs];
h = pie(Pyr);
set(findobj(gca,'type','line'),'linew',2);
% title({'Pyr'});
% set(gca,'FontSize',20);
% set(findobj(h,'type','text'),'fontsize',20);
colormap([1 1 1;0 0 0]);
saveas(gcf,'Fig4_beforeDrugsPyr_Pie.png');

Fig2x2 = figure;
PV = [PV_yesTI_beforeDrugs PV_noTI_beforeDrugs];
h = pie(PV);
set(findobj(gca,'type','line'),'linew',2);
% title({'PV'});
% set(gca,'FontSize',20);
% set(findobj(h,'type','text'),'fontsize',20);
colormap([1 1 1;0 0 0]);
saveas(gcf,'Fig4_beforeDrugsPV_Pie.png');


%% Does or does not exhibit TI PIE CHARTS _afterDrugs
mainFolder = cd('C:\Users\chait\OneDrive\Desktop\SARA\specialCase\recheckingAll_240720\inputFiles'); % CHANGE directory accordingly where the inputFiles folder is
results = readtable('allCellsResultsSummary.xlsx');
cd(mainFolder); %% return to main directory that should be where the outputFiles folder is

Pyr_yesTI_afterDrugs = 0;
Pyr_noTI_afterDrugs = 0;
Pyr_total_Drugs = 0;

PV_yesTI_afterDrugs = 0;
PV_noTI_afterDrugs = 0;
PV_total_Drugs = 0;

for i = 1:height(results);
    if string(results{i,2}) == 'Pyr'
        if string(results{i,9}) == 'afterDrugs'
            Pyr_total_Drugs = Pyr_total_Drugs + 1;
            if string(results{i,48}) == 'yes'
                Pyr_yesTI_afterDrugs = Pyr_yesTI_afterDrugs + 1;
            end
            Pyr_noTI_afterDrugs = Pyr_total_Drugs - Pyr_yesTI_afterDrugs;
        end
    end
    if string(results{i,2}) == 'PV'
        if string(results{i,9}) == 'afterDrugs'
            PV_total_Drugs = PV_total_Drugs + 1;
            if string(results{i,48}) == 'yes'
                PV_yesTI_afterDrugs = PV_yesTI_afterDrugs + 1;
            end
            PV_noTI_afterDrugs = PV_total_Drugs - PV_yesTI_afterDrugs;
        end
    end
end

Fig2x1 = figure;
Pyr = [Pyr_yesTI_afterDrugs Pyr_noTI_afterDrugs];
h = pie(Pyr);
set(findobj(gca,'type','line'),'linew',2);
% title({'Pyr'});
% set(gca,'FontSize',20);
% set(findobj(h,'type','text'),'fontsize',20);
colormap([1 1 1;0 0 0]);
saveas(gcf,'Fig4_afterDrugsPyr_Pie.png');

Fig2x2 = figure;
PV = [PV_yesTI_afterDrugs PV_noTI_afterDrugs];
h = pie(PV);
set(findobj(gca,'type','line'),'linew',2);
% title({'PV'});
% set(gca,'FontSize',20);
% set(findobj(h,'type','text'),'fontsize',20);
colormap([1 1 1;0 0 0]);
saveas(gcf,'Fig4_afterDrugsPV_Pie.png');




%% SI Fig1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Pyr exhibits TI
% 20210119b

% 1 - Extract data from txt file
mainFolder = cd('C:\Users\chait\OneDrive\Desktop\SARA\specialCase\recheckingAll_240720\inputFiles'); % CHANGE directory accordingly where the inputFiles folder is
part1Rec = readtable("SCM_20210119b_1acCL2-3_sePtIr2f_{0}.txt"); %%15,15
part2Rec = readtable("SCM_20210119b_1acCL2-3_sePtIr2f_{1}.txt");
notes = readtable('20210119b_Notes.txt');
selectedIntraSweep = 16
selectedSinSweep = (table2array(notes(17,1)))+1
selectedTISweep = (table2array(notes(9,1)))+1
cd(mainFolder); %% return to main directory that should be where the outputFiles folder is

% 2 - Prepare data
newnames = matlab.lang.makeUniqueStrings([part1Rec.Properties.VariableNames, part2Rec.Properties.VariableNames]);
part1Rec.Properties.VariableNames = newnames(1:numel(part1Rec.Properties.VariableNames));
part2Rec.Properties.VariableNames = newnames(numel(part1Rec.Properties.VariableNames)+1:end);
Rec = [part1Rec part2Rec];
% remove NaN columns
badColumns = isnan(Rec{1,:});
CleanRec = Rec(:,~badColumns);
% segregate Time and Voltage sweeps
Time_Sweeps = CleanRec(:,1:2:end-1);
Voltage_Sweeps = CleanRec(:,2:2:end);

% 3 - Plot traces
% firing pattern --> intracellular stimulation
currentVoltage_Sweep = table2array(Voltage_Sweeps(:,selectedIntraSweep));
currentTime_Sweep = table2array(Time_Sweeps(:,selectedIntraSweep));
[b,a] = butter(3,[100,1000]/(10000/2));
filtVoltage_Sweep = filtfilt(b,a,currentVoltage_Sweep);
[pks,locs] = findpeaks(filtVoltage_Sweep, currentTime_Sweep, 'MinPeakProminence',20, "MinPeakHeight",15);
NumPeaks_Sweeps = length(pks);

Fig1c1 = figure;
x = currentTime_Sweep;
y1 = currentVoltage_Sweep;
plot(x,y1,'black','LineWidth',1);
title({'PYR', 'Intra Stim','exhibits TI'});
xlim([[1.8 2.7]]);
ylim([-130 100]);
xlabel('Time (s)', 'FontSize', 14);
ylabel('Voltage (mV)', 'FontSize', 14);
set(gca,'FontSize',14);
box off;
axis off;
saveas(gcf,'SupplFig1_20210119b_PYR_IntraStim_exhibitsTI.png');

currentVoltage_Sweep = table2array(Voltage_Sweeps(:,selectedSinSweep));
currentTime_Sweep = table2array(Time_Sweeps(:,selectedSinSweep));
[b,a] = butter(3,[100,1000]/(10000/2));
filtVoltage_Sweep = filtfilt(b,a,currentVoltage_Sweep);
[pks,locs] = findpeaks(filtVoltage_Sweep, currentTime_Sweep, 'MinPeakProminence',20, "MinPeakHeight",15);
NumPeaks_Sweeps = length(pks);

Fig1d1 = figure;
x = currentTime_Sweep;
y1 = currentVoltage_Sweep;
plot(x,y1,'black','LineWidth',1);
title({'PYR', 'Sin Stim','exhibits TI'});
x1 = (table2array(notes(20,1))) - 0.2
x2 = x1 + 0.9
xlim([x1 x2]);
ylim([-130 100]);
xlabel('Time (s)', 'FontSize', 14);
ylabel('Voltage (mV)', 'FontSize', 14);
set(gca,'FontSize',14);
box off;
axis off;
saveas(gcf,'SupplFig1_20210119b_PYR_SinStim_exhibitsTI.png');

Fig1d2 = figure;
x = currentTime_Sweep;
y1 = filtVoltage_Sweep;
plot(x,y1,'black','LineWidth',1);
title({'PYR', 'Sin Stim Filt','exhibits TI'});
x1 = (table2array(notes(20,1))) - 0.2
x2 = x1 + 0.9
xlim([x1 x2]);
ylim([-130 100]);
xlabel('Time (s)', 'FontSize', 14);
ylabel('Voltage (mV)', 'FontSize', 14);
box off;
axis off;
saveas(gcf,'SupplFig1_20210119b_PYR_SinStimFilt_exhibitsTI.png');

Fig1d3 = figure;
x = currentTime_Sweep;
y1 = currentVoltage_Sweep;
plot(x,y1,'black','LineWidth',1);
title({'PYR', 'Sin Stim Zoom','exhibits TI'});
xlim([6.22 6.24]);
ylim([-130 100]);
xlabel('Time (s)', 'FontSize', 14);
ylabel('Voltage (mV)', 'FontSize', 14);
set(gca,'FontSize',14);
box off;
axis off;
saveas(gcf,'SupplFig1_20210119b_PYR_SinStimZoom_exhibitsTI.png');

Fig1d4 = figure;
x = currentTime_Sweep;
y1 = filtVoltage_Sweep;
plot(x,y1,'black','LineWidth',1);
title({'PYR', 'Sin Stim Zoom Filt','exhibits TI'});
xlim([6.22 6.24]);
ylim([-130 100]);
xlabel('Time (s)', 'FontSize', 14);
ylabel('Voltage (mV)', 'FontSize', 14);
set(gca,'FontSize',14);
box off;
axis off;
saveas(gcf,'SupplFig1_20210119b_PYR_SinStimZoomFilt_exhibitsTI.png');

% Sweep
currentVoltage_Sweep = table2array(Voltage_Sweeps(:,selectedTISweep));
currentTime_Sweep = table2array(Time_Sweeps(:,selectedTISweep));
[b,a] = butter(3,[100,1000]/(10000/2));
filtVoltage_Sweep = filtfilt(b,a,currentVoltage_Sweep);
[pks,locs] = findpeaks(filtVoltage_Sweep, currentTime_Sweep, 'MinPeakProminence',20, "MinPeakHeight",15);
NumPeaks_Sweeps = length(pks);

Fig1e1 = figure;
x = currentTime_Sweep;
y1 = currentVoltage_Sweep;
plot(x,y1,'black','LineWidth',1);
title({'PYR', 'TI Stim','exhibits TI'});
x1 = (table2array(notes(12,1))) - 0.2
x2 = x1 + 0.9
xlim([x1 x2]);
ylim([-130 100]);
xlabel('Time (s)', 'FontSize', 14);
ylabel('Voltage (mV)', 'FontSize', 14);
set(gca,'FontSize',14);
box off;
axis off;
saveas(gcf,'SupplFig1_20210119b_PYR_TIStim_exhibitsTI.png');

Fig1e2 = figure;
x = currentTime_Sweep;
y1 = filtVoltage_Sweep;
plot(x,y1,'black','LineWidth',1);
title({'PYR', 'TI Stim Filt','exhibits TI'});
x1 = (table2array(notes(12,1))) - 0.2
x2 = x1 + 0.9
xlim([x1 x2]);
ylim([-130 100]);
xlabel('Time (s)', 'FontSize', 14);
ylabel('Voltage (mV)', 'FontSize', 14);
set(gca,'FontSize',14);
box off;
axis off;
saveas(gcf,'SupplFig1_20210119b_PYR_TIStimFilt_exhibitsTI.png');

Fig1e3 = figure;
x = currentTime_Sweep;
y1 = currentVoltage_Sweep;
plot(x,y1,'black','LineWidth',1);
title({'PYR', 'TI Stim Zoom','exhibits TI'});
xlim([6.12 6.14]);
ylim([-130 100]);
xlabel('Time (s)', 'FontSize', 14);
ylabel('Voltage (mV)', 'FontSize', 14);
set(gca,'FontSize',14);
box off;
axis off;
saveas(gcf,'SupplFig1_20210119b_PYR_TIStimZoom_exhibitsTI.png');

Fig1e4 = figure;
x = currentTime_Sweep;
y1 = filtVoltage_Sweep;
plot(x,y1,'black','LineWidth',1);
title({'PYR', 'TI Stim Zoom Filt','exhibits TI'});
xlim([6.12 6.14]);
ylim([-130 100]);
xlabel('Time (s)', 'FontSize', 14);
ylabel('Voltage (mV)', 'FontSize', 14);
set(gca,'FontSize',14);
box off;
axis off;
saveas(gcf,'SupplFig1_20210119b_PYR_TIStimZoomFilt_exhibitsTI.png');


%% Pyr does not exhibit TI
% 20210410a

% 1 - Extract data from txt file
mainFolder = cd('C:\Users\chait\OneDrive\Desktop\SARA\specialCase\recheckingAll_240720\inputFiles'); % CHANGE directory accordingly where the inputFiles folder is
part1Rec = readtable("SCM_20210410a_2aaCL2-3_sePtIr4f_{0}.txt"); %%15,15
part2Rec = readtable("SCM_20210410a_2aaCL2-3_sePtIr4f_{1}.txt");
notes = readtable('20210410a_Notes.txt');
selectedIntraSweep = 16
selectedSinSweep = (table2array(notes(17,1)))+1
selectedTISweep = (table2array(notes(9,1)))+1
cd(mainFolder); %% return to main directory that should be where the outputFiles folder is

% 2 - Prepare data
newnames = matlab.lang.makeUniqueStrings([part1Rec.Properties.VariableNames, part2Rec.Properties.VariableNames]);
part1Rec.Properties.VariableNames = newnames(1:numel(part1Rec.Properties.VariableNames));
part2Rec.Properties.VariableNames = newnames(numel(part1Rec.Properties.VariableNames)+1:end);
Rec = [part1Rec part2Rec];
% remove NaN columns
badColumns = isnan(Rec{1,:});
CleanRec = Rec(:,~badColumns);
% segregate Time and Voltage sweeps
Time_Sweeps = CleanRec(:,1:2:end-1);
Voltage_Sweeps = CleanRec(:,2:2:end);

% 3 - Plot traces
% firing pattern --> intracellular stimulation
currentVoltage_Sweep = table2array(Voltage_Sweeps(:,selectedIntraSweep));
currentTime_Sweep = table2array(Time_Sweeps(:,selectedIntraSweep));
[b,a] = butter(3,[100,1000]/(10000/2));
filtVoltage_Sweep = filtfilt(b,a,currentVoltage_Sweep);
[pks,locs] = findpeaks(filtVoltage_Sweep, currentTime_Sweep, 'MinPeakProminence',20, "MinPeakHeight",15);
NumPeaks_Sweeps = length(pks);

Fig1c1 = figure;
x = currentTime_Sweep;
y1 = currentVoltage_Sweep;
plot(x,y1,'black','LineWidth',1);
title({'PYR', 'Intra Stim','no TI'});
xlim([[1.8 2.7]]);
ylim([-130 100]);
xlabel('Time (s)', 'FontSize', 14);
ylabel('Voltage (mV)', 'FontSize', 14);
set(gca,'FontSize',14);
saveas(gcf,'SupplFig1_20210410a_PYR_IntraStim_noTI_scale.png');

% Sweep
currentVoltage_Sweep = table2array(Voltage_Sweeps(:,selectedSinSweep));
currentTime_Sweep = table2array(Time_Sweeps(:,selectedSinSweep));
[b,a] = butter(3,[100,1000]/(10000/2));
filtVoltage_Sweep = filtfilt(b,a,currentVoltage_Sweep);
[pks,locs] = findpeaks(filtVoltage_Sweep, currentTime_Sweep, 'MinPeakProminence',20, "MinPeakHeight",15);
NumPeaks_Sweeps = length(pks);

Fig1d1 = figure;
x = currentTime_Sweep;
y1 = currentVoltage_Sweep;
plot(x,y1,'black','LineWidth',1);
title({'PYR', 'Sin Stim','no TI'});
x1 = (table2array(notes(20,1))) - 0.2
x2 = x1 + 0.9
xlim([x1 x2]);
ylim([-130 100]);
xlabel('Time (s)', 'FontSize', 14);
ylabel('Voltage (mV)', 'FontSize', 14);
set(gca,'FontSize',14);
saveas(gcf,'SupplFig1_20210410a_PYR_SinStim_noTI_scale.png');

Fig1d2 = figure;
x = currentTime_Sweep;
y1 = filtVoltage_Sweep;
plot(x,y1,'black','LineWidth',1);
title({'PYR', 'Sin Stim Filt','no TI'});
x1 = (table2array(notes(20,1))) - 0.2
x2 = x1 + 0.9
xlim([x1 x2]);
ylim([-130 100]);
xlabel('Time (s)', 'FontSize', 14);
ylabel('Voltage (mV)', 'FontSize', 14);
box off;
axis off;
saveas(gcf,'SupplFig1_20210410a_PYR_SinStimFilt_noTI.png');

Fig1d3 = figure;
x = currentTime_Sweep;
y1 = currentVoltage_Sweep;
plot(x,y1,'black','LineWidth',1);
title({'PYR', 'Sin Stim Zoom','no TI'});
xlim([5.934 5.954]);
ylim([-130 100]);
xlabel('Time (s)', 'FontSize', 14);
ylabel('Voltage (mV)', 'FontSize', 14);
set(gca,'FontSize',14);
box off;
axis off;
saveas(gcf,'SupplFig1_20210410a_PYR_SinStimZoom_noTI.png');

Fig1d4 = figure;
x = currentTime_Sweep;
y1 = filtVoltage_Sweep;
plot(x,y1,'black','LineWidth',1);
title({'PYR', 'Sin Stim Zoom Filt','no TI'});
xlim([5.934 5.954]);
ylim([-130 100]);
xlabel('Time (s)', 'FontSize', 14);
ylabel('Voltage (mV)', 'FontSize', 14);
set(gca,'FontSize',14);
box off;
axis off;
saveas(gcf,'SupplFig1_20210410a_PYR_SinStimZoomFilt_noTI.png');

% Sweep
currentVoltage_Sweep = table2array(Voltage_Sweeps(:,selectedTISweep));
currentTime_Sweep = table2array(Time_Sweeps(:,selectedTISweep));
[b,a] = butter(3,[100,1000]/(10000/2));
filtVoltage_Sweep = filtfilt(b,a,currentVoltage_Sweep);
[pks,locs] = findpeaks(filtVoltage_Sweep, currentTime_Sweep, 'MinPeakProminence',20, "MinPeakHeight",15);
NumPeaks_Sweeps = length(pks);

Fig1e1 = figure;
x = currentTime_Sweep;
y1 = currentVoltage_Sweep;
plot(x,y1,'black','LineWidth',1);
title({'PYR', 'TI Stim','no TI'});
x1 = (table2array(notes(12,1))) - 0.2
x2 = x1 + 0.9
xlim([x1 x2]);
ylim([-130 100]);
xlabel('Time (s)', 'FontSize', 14);
ylabel('Voltage (mV)', 'FontSize', 14);
set(gca,'FontSize',14);
box off;
axis off;
saveas(gcf,'SupplFig1_20210410a_PYR_TIStim_noTI.png');

Fig1e2 = figure;
x = currentTime_Sweep;
y1 = filtVoltage_Sweep;
plot(x,y1,'black','LineWidth',1);
title({'PYR', 'TI Stim Filt','no TI'});
x1 = (table2array(notes(12,1))) - 0.2
x2 = x1 + 0.9
xlim([x1 x2]);
ylim([-130 100]);
xlabel('Time (s)', 'FontSize', 14);
ylabel('Voltage (mV)', 'FontSize', 14);
set(gca,'FontSize',14);
box off;
axis off;
saveas(gcf,'SupplFig1_20210410a_PYR_TIStimFilt_noTI.png');

Fig1e3 = figure;
x = currentTime_Sweep;
y1 = currentVoltage_Sweep;
plot(x,y1,'black','LineWidth',1);
title({'PYR', 'TI Stim Zoom','no TI'});
xlim([6.148 6.168]);
ylim([-130 100]);
xlabel('Time (s)', 'FontSize', 14);
ylabel('Voltage (mV)', 'FontSize', 14);
set(gca,'FontSize',14);
box off;
axis off;
saveas(gcf,'SupplFig1_20210410a_PYR_TIStimZoom_noTI.png');

Fig1e4 = figure;
x = currentTime_Sweep;
y1 = filtVoltage_Sweep;
plot(x,y1,'black','LineWidth',1);
title({'PYR', 'TI Stim Zoom Filt','no TI'});
xlim([6.148 6.168]);
ylim([-130 100]);
xlabel('Time (s)', 'FontSize', 14);
ylabel('Voltage (mV)', 'FontSize', 14);
set(gca,'FontSize',14);
box off;
axis off;
saveas(gcf,'SupplFig1_20210410a_PYR_TIStimZoomFilt_noTI.png');


%% PV does exhibits TI
% 20210518a

% 1 - Extract data from txt file
mainFolder = cd('C:\Users\chait\OneDrive\Desktop\SARA\specialCase\recheckingAll_240720\inputFiles'); % CHANGE directory accordingly where the inputFiles folder is
part1Rec = readtable("SCM_20210518a_3aaCL2-3_seIr-Pt4f_{0}.txt"); %%15,15
part2Rec = readtable("SCM_20210518a_3aaCL2-3_seIr-Pt4f_{1}.txt");
notes = readtable('20210518a_Notes.txt');
selectedIntraSweep = 16
selectedSinSweep = (table2array(notes(17,1)))+1
selectedTISweep = (table2array(notes(9,1)))+1
cd(mainFolder); %% return to main directory that should be where the outputFiles folder is

% 2 - Prepare data
newnames = matlab.lang.makeUniqueStrings([part1Rec.Properties.VariableNames, part2Rec.Properties.VariableNames]);
part1Rec.Properties.VariableNames = newnames(1:numel(part1Rec.Properties.VariableNames));
part2Rec.Properties.VariableNames = newnames(numel(part1Rec.Properties.VariableNames)+1:end);
Rec = [part1Rec part2Rec];
% remove NaN columns
badColumns = isnan(Rec{1,:});
CleanRec = Rec(:,~badColumns);
% segregate Time and Voltage sweeps
Time_Sweeps = CleanRec(:,1:2:end-1);
Voltage_Sweeps = CleanRec(:,2:2:end);

% 3 - Plot traces
% firing pattern --> intracellular stimulation
currentVoltage_Sweep = table2array(Voltage_Sweeps(:,selectedIntraSweep));
currentTime_Sweep = table2array(Time_Sweeps(:,selectedIntraSweep));
[b,a] = butter(3,[100,1000]/(10000/2));
filtVoltage_Sweep = filtfilt(b,a,currentVoltage_Sweep);
[pks,locs] = findpeaks(filtVoltage_Sweep, currentTime_Sweep, 'MinPeakProminence',20, "MinPeakHeight",15);
NumPeaks_Sweeps = length(pks);

Fig1c1 = figure;
x = currentTime_Sweep;
y1 = currentVoltage_Sweep;
plot(x,y1,'black','LineWidth',1);
title({'PV', 'Intra Stim','exhibits TI'});
xlim([[1.8 2.7]]);
ylim([-130 100]);
xlabel('Time (s)', 'FontSize', 14);
ylabel('Voltage (mV)', 'FontSize', 14);
set(gca,'FontSize',14);
box off;
axis off;
saveas(gcf,'SupplFig2_20210518a_PV_IntraStim_exhibitsTI.png');

% Sweep
currentVoltage_Sweep = table2array(Voltage_Sweeps(:,selectedSinSweep));
currentTime_Sweep = table2array(Time_Sweeps(:,selectedSinSweep));
[b,a] = butter(3,[100,1000]/(10000/2));
filtVoltage_Sweep = filtfilt(b,a,currentVoltage_Sweep);
[pks,locs] = findpeaks(filtVoltage_Sweep, currentTime_Sweep, 'MinPeakProminence',20, "MinPeakHeight",15);
NumPeaks_Sweeps = length(pks);

Fig1d1 = figure;
x = currentTime_Sweep;
y1 = currentVoltage_Sweep;
plot(x,y1,'black','LineWidth',1);
title({'PV', 'Sin Stim','exhibits TI'});
x1 = (table2array(notes(20,1))) - 0.2
x2 = x1 + 0.9
xlim([x1 x2]);
ylim([-130 100]);
xlabel('Time (s)', 'FontSize', 14);
ylabel('Voltage (mV)', 'FontSize', 14);
set(gca,'FontSize',14);
box off;
axis off;
saveas(gcf,'SupplFig2_20210518a_PV_SinStim_exhibitsTI.png');

Fig1d2 = figure;
x = currentTime_Sweep;
y1 = filtVoltage_Sweep;
plot(x,y1,'black','LineWidth',1);
title({'PV', 'Sin Stim Filt','exhibits TI'});
x1 = (table2array(notes(20,1))) - 0.2
x2 = x1 + 0.9
xlim([x1 x2]);
ylim([-130 100]);
xlabel('Time (s)', 'FontSize', 14);
ylabel('Voltage (mV)', 'FontSize', 14);
box off;
axis off;
saveas(gcf,'SupplFig2_20210518a_PV_SinStimFilt_exhibitsTI.png');

Fig1d3 = figure;
x = currentTime_Sweep;
y1 = currentVoltage_Sweep;
plot(x,y1,'black','LineWidth',1);
title({'PV', 'Sin Stim Zoom','exhibits TI'});
xlim([6.366 6.386]);
ylim([-130 100]);
xlabel('Time (s)', 'FontSize', 14);
ylabel('Voltage (mV)', 'FontSize', 14);
set(gca,'FontSize',14);
box off;
axis off;
saveas(gcf,'SupplFig2_20210518a_PV_SinStimZoom_exhibitsTI.png');

Fig1d4 = figure;
x = currentTime_Sweep;
y1 = filtVoltage_Sweep;
plot(x,y1,'black','LineWidth',1);
title({'PV', 'Sin Stim Zoom Filt','exhibits TI'});
xlim([6.366 6.386]);
ylim([-130 100]);
xlabel('Time (s)', 'FontSize', 14);
ylabel('Voltage (mV)', 'FontSize', 14);
set(gca,'FontSize',14);
box off;
axis off;
saveas(gcf,'SupplFig2_20210518a_PV_SinStimZoomFilt_exhibitsTI.png');

% Sweep
currentVoltage_Sweep = table2array(Voltage_Sweeps(:,selectedTISweep));
currentTime_Sweep = table2array(Time_Sweeps(:,selectedTISweep));
[b,a] = butter(3,[100,1000]/(10000/2));
filtVoltage_Sweep = filtfilt(b,a,currentVoltage_Sweep);
[pks,locs] = findpeaks(filtVoltage_Sweep, currentTime_Sweep, 'MinPeakProminence',20, "MinPeakHeight",15);
NumPeaks_Sweeps = length(pks);

Fig1e1 = figure;
x = currentTime_Sweep;
y1 = currentVoltage_Sweep;
plot(x,y1,'black','LineWidth',1);
title({'PV', 'TI Stim','exhibits TI'});
x1 = (table2array(notes(12,1))) - 0.2
x2 = x1 + 0.9
xlim([x1 x2]);
ylim([-130 100]);
xlabel('Time (s)', 'FontSize', 14);
ylabel('Voltage (mV)', 'FontSize', 14);
set(gca,'FontSize',14);
box off;
axis off;
saveas(gcf,'SupplFig2_20210518a_PV_TIStim_exhibitsTI.png');

Fig1e2 = figure;
x = currentTime_Sweep;
y1 = filtVoltage_Sweep;
plot(x,y1,'black','LineWidth',1);
title({'PV', 'TI Stim Filt','exhibits TI'});
x1 = (table2array(notes(12,1))) - 0.2
x2 = x1 + 0.9
xlim([x1 x2]);
ylim([-130 100]);
xlabel('Time (s)', 'FontSize', 14);
ylabel('Voltage (mV)', 'FontSize', 14);
set(gca,'FontSize',14);
box off;
axis off;
saveas(gcf,'SupplFig2_20210518a_PV_TIStimFilt_exhibitsTI.png');

Fig1e3 = figure;
x = currentTime_Sweep;
y1 = currentVoltage_Sweep;
plot(x,y1,'black','LineWidth',1);
title({'PV', 'TI Stim Zoom','exhibits TI'});
xlim([6.572 6.592]);
ylim([-130 100]);
xlabel('Time (s)', 'FontSize', 14);
ylabel('Voltage (mV)', 'FontSize', 14);
set(gca,'FontSize',14);
box off;
axis off;
saveas(gcf,'SupplFig2_20210518a_PV_TIStimZoom_exhibitsTI.png');

Fig1e4 = figure;
x = currentTime_Sweep;
y1 = filtVoltage_Sweep;
plot(x,y1,'black','LineWidth',1);
title({'PV', 'TI Stim Zoom Filt','exhibits TI'});
xlim([6.572 6.592]);
ylim([-130 100]);
xlabel('Time (s)', 'FontSize', 14);
ylabel('Voltage (mV)', 'FontSize', 14);
set(gca,'FontSize',14);
box off;
axis off;
saveas(gcf,'SupplFig2_20210518a_PV_TIStimZoomFilt_exhibitsTI.png');




%% PV does not exhibit TI
% 20210517a

% 1 - Extract data from txt file
mainFolder = cd('C:\Users\chait\OneDrive\Desktop\SARA\specialCase\recheckingAll_240720\inputFiles'); % CHANGE directory accordingly where the inputFiles folder is
part1Rec = readtable("SCM_20210517a_3aaCL2-3_seIr-Pt4f_{0}.txt"); %%15,15
part2Rec = readtable("SCM_20210517a_3aaCL2-3_seIr-Pt4f_{1}.txt");
notes = readtable('20210517a_Notes.txt');
selectedIntraSweep = 16
selectedSinSweep = (table2array(notes(17,1)))+1
selectedTISweep = (table2array(notes(9,1)))+1
cd(mainFolder); %% return to main directory that should be where the outputFiles folder is

% 2 - Prepare data
newnames = matlab.lang.makeUniqueStrings([part1Rec.Properties.VariableNames, part2Rec.Properties.VariableNames]);
part1Rec.Properties.VariableNames = newnames(1:numel(part1Rec.Properties.VariableNames));
part2Rec.Properties.VariableNames = newnames(numel(part1Rec.Properties.VariableNames)+1:end);
Rec = [part1Rec part2Rec];
% remove NaN columns
badColumns = isnan(Rec{1,:});
CleanRec = Rec(:,~badColumns);
% segregate Time and Voltage sweeps
Time_Sweeps = CleanRec(:,1:2:end-1);
Voltage_Sweeps = CleanRec(:,2:2:end);

% 3 - Plot traces
% firing pattern --> intracellular stimulation
currentVoltage_Sweep = table2array(Voltage_Sweeps(:,selectedIntraSweep));
currentTime_Sweep = table2array(Time_Sweeps(:,selectedIntraSweep));
[b,a] = butter(3,[100,1000]/(10000/2));
filtVoltage_Sweep = filtfilt(b,a,currentVoltage_Sweep);
[pks,locs] = findpeaks(filtVoltage_Sweep, currentTime_Sweep, 'MinPeakProminence',20, "MinPeakHeight",15);
NumPeaks_Sweeps = length(pks);

Fig1c1 = figure;
x = currentTime_Sweep;
y1 = currentVoltage_Sweep;
plot(x,y1,'black','LineWidth',1);
title({'PV', 'Intra Stim','no TI'});
xlim([[1.8 2.7]]);
ylim([-130 100]);
xlabel('Time (s)', 'FontSize', 14);
ylabel('Voltage (mV)', 'FontSize', 14);
set(gca,'FontSize',14);
box off;
axis off;
saveas(gcf,'SupplFig2_20210517a_PV_IntraStim_noTI.png');

% Sweep
currentVoltage_Sweep = table2array(Voltage_Sweeps(:,selectedSinSweep));
currentTime_Sweep = table2array(Time_Sweeps(:,selectedSinSweep));
[b,a] = butter(3,[100,1000]/(10000/2));
filtVoltage_Sweep = filtfilt(b,a,currentVoltage_Sweep);
[pks,locs] = findpeaks(filtVoltage_Sweep, currentTime_Sweep, 'MinPeakProminence',20, "MinPeakHeight",15);
NumPeaks_Sweeps = length(pks);

Fig1d1 = figure;
x = currentTime_Sweep;
y1 = currentVoltage_Sweep;
plot(x,y1,'black','LineWidth',1);
title({'PV', 'Sin Stim','no TI'});
x1 = (table2array(notes(20,1))) - 0.2
x2 = x1 + 0.9
xlim([x1 x2]);
ylim([-130 100]);
xlabel('Time (s)', 'FontSize', 14);
ylabel('Voltage (mV)', 'FontSize', 14);
set(gca,'FontSize',14);
box off;
axis off;
saveas(gcf,'SupplFig2_20210517a_PV_SinStim_noTI.png');

Fig1d2 = figure;
x = currentTime_Sweep;
y1 = filtVoltage_Sweep;
plot(x,y1,'black','LineWidth',1);
title({'PV', 'Sin Stim Filt','no TI'});
x1 = (table2array(notes(20,1))) - 0.2
x2 = x1 + 0.9
xlim([x1 x2]);
ylim([-130 100]);
xlabel('Time (s)', 'FontSize', 14);
ylabel('Voltage (mV)', 'FontSize', 14);
box off;
axis off;
saveas(gcf,'SupplFig2_20210517a_PV_SinStimFilt_noTI.png');

Fig1d3 = figure;
x = currentTime_Sweep;
y1 = currentVoltage_Sweep;
plot(x,y1,'black','LineWidth',1);
title({'PV', 'Sin Stim Zoom','no TI'});
xlim([6.064 6.084]);
ylim([-130 100]);
xlabel('Time (s)', 'FontSize', 14);
ylabel('Voltage (mV)', 'FontSize', 14);
set(gca,'FontSize',14);
box off;
axis off;
saveas(gcf,'SupplFig2_20210517a_PV_SinStimZoom_noTI.png');

Fig1d4 = figure;
x = currentTime_Sweep;
y1 = filtVoltage_Sweep;
plot(x,y1,'black','LineWidth',1);
title({'PV', 'Sin Stim Zoom Filt','no TI'});
xlim([6.064 6.084]);
ylim([-130 100]);
xlabel('Time (s)', 'FontSize', 14);
ylabel('Voltage (mV)', 'FontSize', 14);
set(gca,'FontSize',14);
box off;
axis off;
saveas(gcf,'SupplFig2_20210517a_PV_SinStimZoomFilt_noTI.png');

% Sweep
currentVoltage_Sweep = table2array(Voltage_Sweeps(:,selectedTISweep));
currentTime_Sweep = table2array(Time_Sweeps(:,selectedTISweep));
[b,a] = butter(3,[100,1000]/(10000/2));
filtVoltage_Sweep = filtfilt(b,a,currentVoltage_Sweep);
[pks,locs] = findpeaks(filtVoltage_Sweep, currentTime_Sweep, 'MinPeakProminence',20, "MinPeakHeight",15);
NumPeaks_Sweeps = length(pks);

Fig1e1 = figure;
x = currentTime_Sweep;
y1 = currentVoltage_Sweep;
plot(x,y1,'black','LineWidth',1);
title({'PV', 'TI Stim','no TI'});
x1 = (table2array(notes(12,1))) - 0.2
x2 = x1 + 0.9
xlim([x1 x2]);
ylim([-130 100]);
xlabel('Time (s)', 'FontSize', 14);
ylabel('Voltage (mV)', 'FontSize', 14);
set(gca,'FontSize',14);
box off;
axis off;
saveas(gcf,'SupplFig2_20210517a_PV_TIStim_noTI.png');

Fig1e2 = figure;
x = currentTime_Sweep;
y1 = filtVoltage_Sweep;
plot(x,y1,'black','LineWidth',1);
title({'PV', 'TI Stim Filt','no TI'});
x1 = (table2array(notes(12,1))) - 0.2
x2 = x1 + 0.9
xlim([x1 x2]);
ylim([-130 100]);
xlabel('Time (s)', 'FontSize', 14);
ylabel('Voltage (mV)', 'FontSize', 14);
set(gca,'FontSize',14);
box off;
axis off;
saveas(gcf,'SupplFig2_20210517a_PV_TIStimFilt_noTI.png');

Fig1e3 = figure;
x = currentTime_Sweep;
y1 = currentVoltage_Sweep;
plot(x,y1,'black','LineWidth',1);
title({'PV', 'TI Stim Zoom','no TI'});
xlim([6.228 6.248]);
ylim([-130 100]);
xlabel('Time (s)', 'FontSize', 14);
ylabel('Voltage (mV)', 'FontSize', 14);
set(gca,'FontSize',14);
box off;
axis off;
saveas(gcf,'SupplFig2_20210517a_PV_TIStimZoom_noTI.png');

Fig1e4 = figure;
x = currentTime_Sweep;
y1 = filtVoltage_Sweep;
plot(x,y1,'black','LineWidth',1);
title({'PV', 'TI Stim Zoom Filt','no TI'});
xlim([6.228 6.248]);
ylim([-130 100]);
xlabel('Time (s)', 'FontSize', 14);
ylabel('Voltage (mV)', 'FontSize', 14);
set(gca,'FontSize',14);
box off;
axis off;
saveas(gcf,'SupplFig2_20210517a_PV_TIStimZoomFilt_noTI.png');


%% SI Fig3 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
mainFolder = cd('C:\Users\chait\OneDrive\Desktop\SARA\specialCase\recheckingAll_240720\inputFiles'); % CHANGE directory accordingly where the inputFiles folder is
results = readtable('allCellsResultsSummary.xlsx');
cd(mainFolder); %% return to main directory that should be where the outputFiles folder is

%% for 2kHz
Pyr_2kHz_I_sin = [];
Pyr_2kHz_I_mod = [];
PV_2kHz_I_sin = [];
PV_2kHz_I_mod = [];

Pyr_2kHz_I_sin_d = [];
Pyr_2kHz_I_mod_d = [];
PV_2kHz_I_sin_d = [];
PV_2kHz_I_mod_d = [];

abc = 3; % constant
for i = 1:height(results)

    % IF Pyr...
    if string(results{i,2}) == 'Pyr' && (string(results{i,6}) == '2' || string(results{i,5}) == '2')
        if string(results{i,47}) == 'yes'
            Pyr_2kHz_I_mod = vertcat(Pyr_2kHz_I_mod,results{i,28-1});
            Pyr_2kHz_I_mod_d = vertcat(Pyr_2kHz_I_mod_d,results{i,abc});
            if string(results{i,49}) == 'sin'
                Pyr_2kHz_I_sin = vertcat(Pyr_2kHz_I_sin,str2double(results{i,59-1}{1}));
                Pyr_2kHz_I_sin_d = vertcat(Pyr_2kHz_I_sin_d,results{i,abc});
            end
        elseif string(results{i,47}) == 'no'
            Pyr_2kHz_I_sin = vertcat(Pyr_2kHz_I_sin,results{i,28-1});
            Pyr_2kHz_I_sin_d = vertcat(Pyr_2kHz_I_sin_d,results{i,abc});
            if string(results{i,49}) == 'mod'
                Pyr_2kHz_I_mod = vertcat(Pyr_2kHz_I_mod,str2double(results{i,59-1}{1}));
                Pyr_2kHz_I_mod_d = vertcat(Pyr_2kHz_I_mod_d,results{i,abc});
            elseif string(results{i,49}) == 'same'
                Pyr_2kHz_I_mod = vertcat(Pyr_2kHz_I_mod,results{i,28-1});
                Pyr_2kHz_I_mod_d = vertcat(Pyr_2kHz_I_mod_d,results{i,abc});
            end
        end
    end

    % IF PV...
    if string(results{i,2}) == 'PV' && (string(results{i,6}) == '2' || string(results{i,5}) == '2')
        if string(results{i,47}) == 'yes'
            PV_2kHz_I_mod = vertcat(PV_2kHz_I_mod,results{i,28-1});
            PV_2kHz_I_mod_d = vertcat(PV_2kHz_I_mod_d,results{i,abc});
            if string(results{i,49}) == 'sin'
                PV_2kHz_I_sin = vertcat(PV_2kHz_I_sin,str2double(results{i,59-1}{1}));
                PV_2kHz_I_sin_d = vertcat(PV_2kHz_I_sin_d,results{i,abc});
            end
        elseif string(results{i,47}) == 'no'
            PV_2kHz_I_sin = vertcat(PV_2kHz_I_sin,results{i,28-1});
            PV_2kHz_I_sin_d = vertcat(PV_2kHz_I_sin_d,results{i,abc});
            if string(results{i,49}) == 'mod'
                PV_2kHz_I_mod = vertcat(PV_2kHz_I_mod,str2double(results{i,59-1}{1}));
                PV_2kHz_I_mod_d = vertcat(PV_2kHz_I_mod_d,results{i,abc});
            elseif string(results{i,49}) == 'same'
                PV_2kHz_I_mod = vertcat(PV_2kHz_I_mod,results{i,28-1});
                PV_2kHz_I_mod_d = vertcat(PV_2kHz_I_mod_d,results{i,abc});
            end
        end
    end
end

Pyr_2kHz_density_sin = [];
for iii = 1:length(Pyr_2kHz_I_sin)
    mainFolder = cd('C:\Users\chait\OneDrive\Desktop\SARA\specialCase\recheckingAll_240720\inputFiles'); % CHANGE directory accordingly where the inputFiles folder is
    norm = load("probeElectricField.mat");
    cd(mainFolder); %% return to main directory that should be where the outputFiles folder is
    norm = norm.probeconcentricHD;
    norm_z = norm(norm(:,3)==0,:);
    distance_N = norm_z(:,2);
    distance_mm = distance_N*10^3;
    intensity = norm_z(:,8);
    norm_intensity = 1e06; % intensity(distance_mm==0.1);
    distance = Pyr_2kHz_I_sin_d(iii)
    distance_given = distance/1000;
    for ii=1:length(distance_given)
        [dump,idx] = min((distance_mm-distance_given(ii)).^2);
        norm_amp(ii) = intensity(idx)/norm_intensity;
    end
    I = Pyr_2kHz_I_sin(iii)
    densityValue = I*norm_amp(1,1)
    Pyr_2kHz_density_sin = vertcat(Pyr_2kHz_density_sin, densityValue)
end

Pyr_2kHz_density_mod = [];
for iii = 1:length(Pyr_2kHz_I_mod)
    mainFolder = cd('C:\Users\chait\OneDrive\Desktop\SARA\specialCase\recheckingAll_240720\inputFiles'); % CHANGE directory accordingly where the inputFiles folder is
    norm = load("probeElectricField.mat");
    cd(mainFolder); %% return to main directory that should be where the outputFiles folder is
    norm = norm.probeconcentricHD;
    norm_z = norm(norm(:,3)==0,:);
    distance_N = norm_z(:,2);
    distance_mm = distance_N*10^3;
    intensity = norm_z(:,8);
   norm_intensity = 1e06; % intensity(distance_mm==0.1);
    distance = Pyr_2kHz_I_mod_d(iii)
    distance_given = distance/1000;
    for ii=1:length(distance_given)
        [dump,idx] = min((distance_mm-distance_given(ii)).^2);
        norm_amp(ii) = intensity(idx)/norm_intensity;
    end
    I = Pyr_2kHz_I_mod(iii)
    densityValue = I*norm_amp(1,1)
    Pyr_2kHz_density_mod = vertcat(Pyr_2kHz_density_mod, densityValue)
end

PV_2kHz_density_sin = [];
for iii = 1:length(PV_2kHz_I_sin)
    mainFolder = cd('C:\Users\chait\OneDrive\Desktop\SARA\specialCase\recheckingAll_240720\inputFiles'); % CHANGE directory accordingly where the inputFiles folder is
    norm = load("probeElectricField.mat");
    cd(mainFolder); %% return to main directory that should be where the outputFiles folder is
    norm = norm.probeconcentricHD;
    norm_z = norm(norm(:,3)==0,:);
    distance_N = norm_z(:,2);
    distance_mm = distance_N*10^3;
    intensity = norm_z(:,8);
   norm_intensity = 1e06; % intensity(distance_mm==0.1);
    distance = PV_2kHz_I_sin_d(iii)
    distance_given = distance/1000;
    for ii=1:length(distance_given)
        [dump,idx] = min((distance_mm-distance_given(ii)).^2);
        norm_amp(ii) = intensity(idx)/norm_intensity;
    end
    I = PV_2kHz_I_sin(iii)
    densityValue = I*norm_amp(1,1)
    PV_2kHz_density_sin = vertcat(PV_2kHz_density_sin, densityValue)
end

PV_2kHz_density_mod = [];
for iii = 1:length(PV_2kHz_I_mod)
    mainFolder = cd('C:\Users\chait\OneDrive\Desktop\SARA\specialCase\recheckingAll_240720\inputFiles'); % CHANGE directory accordingly where the inputFiles folder is
    norm = load("probeElectricField.mat");
    cd(mainFolder); %% return to main directory that should be where the outputFiles folder is
    norm = norm.probeconcentricHD;
    norm_z = norm(norm(:,3)==0,:);
    distance_N = norm_z(:,2);
    distance_mm = distance_N*10^3;
    intensity = norm_z(:,8);
   norm_intensity = 1e06; % intensity(distance_mm==0.1);
    distance = PV_2kHz_I_mod_d(iii)
    distance_given = distance/1000;
    for ii=1:length(distance_given)
        [dump,idx] = min((distance_mm-distance_given(ii)).^2);
        norm_amp(ii) = intensity(idx)/norm_intensity;
    end
    I = PV_2kHz_I_mod(iii)
    densityValue = I*norm_amp(1,1)
    PV_2kHz_density_mod = vertcat(PV_2kHz_density_mod, densityValue)
end

% plots
for i = 1:length(Pyr_2kHz_I_mod)
    Pyr_2kHz_mod(1,i) = 1;
end

for i = 1:length(PV_2kHz_I_sin)
    PV_2kHz_sin(1,i) = 1;
end

for i = 1:length(PV_2kHz_I_mod)
    PV_2kHz_mod(1,i) = 1;
end

% edits for electric field calculation
Pyr_2kHz_density_mod = Pyr_2kHz_density_mod*0.33;
fig3a = figure;
boxes = boxchart(Pyr_2kHz_density_mod)
set(boxes, 'BoxFaceColor', 'none', 'BoxEdgeColor', 'k', 'MarkerStyle','+','MarkerColor','r');
hold on;
ylabel({'Pyr mod', 'Electric Field (V/m)'}, 'FontSize', 18);
ylim([0 290]);
set(gca,'FontSize',18, 'LineWidth',2);
for n=1:max(unique(Pyr_2kHz_mod))
    hs = scatter(ones(sum(Pyr_2kHz_mod==n),1) + n-1, Pyr_2kHz_density_mod(Pyr_2kHz_mod == n),"filled",'jitter','on','JitterAmount',0.1);
    hs.MarkerFaceAlpha = 0.5;
    hs.MarkerFaceColor = 'b'
end
box off
saveas(gcf,'SupplFig3_Pyr_2kHz_I_mod_elecF.tif')
saveas(gcf,'SupplFig3_Pyr_2kHz_I_mod_elecF.fig')

% edits for electric field calculation
PV_2kHz_density_sin = PV_2kHz_density_sin*0.33;
fig3a = figure;
boxes = boxchart(PV_2kHz_density_sin)
set(boxes, 'BoxFaceColor', 'none', 'BoxEdgeColor', 'k', 'MarkerStyle','+','MarkerColor','r');
hold on;
ylabel({'PV sin', 'Electric Field (V/m)'}, 'FontSize', 18);
ylim([0 290]);
set(gca,'FontSize',18, 'LineWidth',2);
for n=1:max(unique(PV_2kHz_sin))
    hs = scatter(ones(sum(PV_2kHz_sin==n),1) + n-1, PV_2kHz_density_sin(PV_2kHz_sin == n),"filled",'jitter','on','JitterAmount',0.1);
    hs.MarkerFaceAlpha = 0.5;
    hs.MarkerFaceColor = 'm'
end
box off
saveas(gcf,'SupplFig3_PV_2kHz_I_sin_elecF.tif')
saveas(gcf,'SupplFig3_PV_2kHz_I_sin_elecF.fig')

% edits for electric field calculation
PV_2kHz_density_mod = PV_2kHz_density_mod*0.33;
fig3a = figure;
boxes = boxchart(PV_2kHz_density_mod)
set(boxes, 'BoxFaceColor', 'none', 'BoxEdgeColor', 'k', 'MarkerStyle','+','MarkerColor','r');
hold on;
ylabel({'PV mod', 'Electric Field (V/m)'}, 'FontSize', 18);
ylim([0 290]);
set(gca,'FontSize',18, 'LineWidth',2);
for n=1:max(unique(PV_2kHz_mod))
    hs = scatter(ones(sum(PV_2kHz_mod==n),1) + n-1, PV_2kHz_density_mod(PV_2kHz_mod == n),"filled",'jitter','on','JitterAmount',0.1);
    hs.MarkerFaceAlpha = 0.5;
    hs.MarkerFaceColor = 'm'
end
box off
saveas(gcf,'SupplFig3_PV_2kHz_I_mod_elecF.tif')
saveas(gcf,'SupplFig3_PV_2kHz_I_mod_elecF.fig')


%% for 4kHz

Pyr_4kHz_I_sin = [];
Pyr_4kHz_I_mod = [];
PV_4kHz_I_sin = [];
PV_4kHz_I_mod = [];

Pyr_4kHz_I_sin_d = [];
Pyr_4kHz_I_mod_d = [];
PV_4kHz_I_sin_d = [];
PV_4kHz_I_mod_d = [];

for i = 1:height(results);

    % IF Pyr...
    if string(results{i,2}) == 'Pyr' && string(results{i,8}) == 'NaN' && (string(results{i,6}) == '4' || string(results{i,5}) == '4')
        if string(results{i,48}) == 'yes'
            Pyr_4kHz_I_mod = vertcat(Pyr_4kHz_I_mod,results{i,45});
            Pyr_4kHz_I_mod_d = vertcat(Pyr_4kHz_I_mod_d,results{i,abc});
            if string(results{i,60}) == 'sin'
                Pyr_4kHz_I_sin = vertcat(Pyr_4kHz_I_sin,str2double(results{i,69}{1}));
                Pyr_4kHz_I_sin_d = vertcat(Pyr_4kHz_I_sin_d,results{i,abc});
            end
        elseif string(results{i,48}) == 'no'
            Pyr_4kHz_I_sin = vertcat(Pyr_4kHz_I_sin,results{i,45});
            Pyr_4kHz_I_sin_d = vertcat(Pyr_4kHz_I_sin_d,results{i,abc});
            if string(results{i,60}) == 'mod'
                Pyr_4kHz_I_mod = vertcat(Pyr_4kHz_I_mod,str2double(results{i,69}{1}));
                Pyr_4kHz_I_mod_d = vertcat(Pyr_4kHz_I_mod_d,results{i,abc});
            elseif string(results{i,60}) == 'same'
                Pyr_4kHz_I_mod = vertcat(Pyr_4kHz_I_mod,results{i,45});
                Pyr_4kHz_I_mod_d = vertcat(Pyr_4kHz_I_mod_d,results{i,abc});
            end
        end
    end

    % IF PV...
    if string(results{i,2}) == 'PV' && string(results{i,8}) == 'NaN' && (string(results{i,6}) == '4' || string(results{i,5}) == '4')
        if string(results{i,48}) == 'yes'
            PV_4kHz_I_mod = vertcat(PV_4kHz_I_mod,results{i,45});
            PV_4kHz_I_mod_d = vertcat(PV_4kHz_I_mod_d,results{i,abc});
            if string(results{i,60}) == 'sin'
                PV_4kHz_I_sin = vertcat(PV_4kHz_I_sin,str2double(results{i,69}{1}));
                PV_4kHz_I_sin_d = vertcat(PV_4kHz_I_sin_d,results{i,abc});
            end
        elseif string(results{i,48}) == 'no'
            PV_4kHz_I_sin = vertcat(PV_4kHz_I_sin,results{i,45});
            PV_4kHz_I_sin_d = vertcat(PV_4kHz_I_sin_d,results{i,abc});
            if string(results{i,60}) == 'mod'
                PV_4kHz_I_mod = vertcat(PV_4kHz_I_mod,str2double(results{i,69}{1}));
                PV_4kHz_I_mod_d = vertcat(PV_4kHz_I_mod_d,results{i,abc});
            elseif string(results{i,60}) == 'same'
                PV_4kHz_I_mod = vertcat(PV_4kHz_I_mod,results{i,45});
                PV_4kHz_I_mod_d = vertcat(PV_4kHz_I_mod_d,results{i,abc});
            end
        end
    end
end

Pyr_4kHz_density_sin = [];
for iii = 1:length(Pyr_4kHz_I_sin)
    mainFolder = cd('C:\Users\chait\OneDrive\Desktop\SARA\specialCase\recheckingAll_240720\inputFiles'); % CHANGE directory accordingly where the inputFiles folder is
    norm = load("probeElectricField.mat");
    cd(mainFolder); %% return to main directory that should be where the outputFiles folder is
    norm = norm.probeconcentricHD;
    norm_z = norm(norm(:,3)==0,:);
    distance_N = norm_z(:,2);
    distance_mm = distance_N*10^3;
    intensity = norm_z(:,8);
    norm_intensity = 1e06; % intensity(distance_mm==0.1);
    distance = Pyr_4kHz_I_sin_d(iii)
    distance_given = distance/1000;
    for ii=1:length(distance_given)
        [dump,idx] = min((distance_mm-distance_given(ii)).^2);
        norm_amp(ii) = intensity(idx)/norm_intensity;
    end
    I = Pyr_4kHz_I_sin(iii)
    densityValue = I*norm_amp(1,1)
    Pyr_4kHz_density_sin = vertcat(Pyr_4kHz_density_sin, densityValue)
    % mainFolder = cd('C:\Users\chait\OneDrive\Desktop\SARA\specialCase\TI_Paper_AllData_InputNotes'); %change directory 220925
    % mainFolder = cd('C:\Users\chait\OneDrive\Desktop\SARA\specialCase\TI_Paper_AllData_InputNotes_beforeAfterDrugsSelection'); %change directory
end

Pyr_4kHz_density_mod = [];
for iii = 1:length(Pyr_4kHz_I_mod)
    mainFolder = cd('C:\Users\chait\OneDrive\Desktop\SARA\specialCase\recheckingAll_240720\inputFiles'); % CHANGE directory accordingly where the inputFiles folder is
    norm = load("probeElectricField.mat");
    cd(mainFolder); %% return to main directory that should be where the outputFiles folder is
    norm = norm.probeconcentricHD;
    norm_z = norm(norm(:,3)==0,:);
    distance_N = norm_z(:,2);
    distance_mm = distance_N*10^3;
    intensity = norm_z(:,8);
    norm_intensity = 1e06; % intensity(distance_mm==0.1);
    distance = Pyr_4kHz_I_mod_d(iii)
    distance_given = distance/1000;
    for ii=1:length(distance_given)
        [dump,idx] = min((distance_mm-distance_given(ii)).^2);
        norm_amp(ii) = intensity(idx)/norm_intensity;
    end
    I = Pyr_4kHz_I_mod(iii)
    densityValue = I*norm_amp(1,1)
    Pyr_4kHz_density_mod = vertcat(Pyr_4kHz_density_mod, densityValue)
end

PV_4kHz_density_sin = [];
for iii = 1:length(PV_4kHz_I_sin)
    mainFolder = cd('C:\Users\chait\OneDrive\Desktop\SARA\specialCase\recheckingAll_240720\inputFiles'); % CHANGE directory accordingly where the inputFiles folder is
    norm = load("probeElectricField.mat");
    cd(mainFolder); %% return to main directory that should be where the outputFiles folder is
    norm = norm.probeconcentricHD;
    norm_z = norm(norm(:,3)==0,:);
    distance_N = norm_z(:,2);
    distance_mm = distance_N*10^3;
    intensity = norm_z(:,8);
    norm_intensity = 1e06; % intensity(distance_mm==0.1);
    distance = PV_4kHz_I_sin_d(iii)
    distance_given = distance/1000;
    for ii=1:length(distance_given)
        [dump,idx] = min((distance_mm-distance_given(ii)).^2);
        norm_amp(ii) = intensity(idx)/norm_intensity;
    end
    I = PV_4kHz_I_sin(iii)
    densityValue = I*norm_amp(1,1)
    PV_4kHz_density_sin = vertcat(PV_4kHz_density_sin, densityValue)
end

PV_4kHz_density_mod = [];
for iii = 1:length(PV_4kHz_I_mod)
    mainFolder = cd('C:\Users\chait\OneDrive\Desktop\SARA\specialCase\recheckingAll_240720\inputFiles'); % CHANGE directory accordingly where the inputFiles folder is
    norm = load("probeElectricField.mat");
    cd(mainFolder); %% return to main directory that should be where the outputFiles folder is
    norm = norm.probeconcentricHD;
    norm_z = norm(norm(:,3)==0,:);
    distance_N = norm_z(:,2);
    distance_mm = distance_N*10^3;
    intensity = norm_z(:,8);
    norm_intensity = 1e06; % intensity(distance_mm==0.1);
    distance = PV_4kHz_I_mod_d(iii)
    distance_given = distance/1000;
    for ii=1:length(distance_given)
        [dump,idx] = min((distance_mm-distance_given(ii)).^2);
        norm_amp(ii) = intensity(idx)/norm_intensity;
    end
    I = PV_4kHz_I_mod(iii)
    densityValue = I*norm_amp(1,1)
    PV_4kHz_density_mod = vertcat(PV_4kHz_density_mod, densityValue)
end

for i = 1:length(Pyr_4kHz_I_mod)
    Pyr_4kHz_mod(1,i) = 1;
end

for i = 1:length(PV_4kHz_I_sin)
    PV_4kHz_sin(1,i) = 1;
end

for i = 1:length(PV_4kHz_I_mod)
    PV_4kHz_mod(1,i) = 1;
end

% edits for electric field calculation
Pyr_4kHz_density_mod = Pyr_4kHz_density_mod*0.33;
fig2 = figure;
boxes = boxchart(Pyr_4kHz_density_mod)
set(boxes, 'BoxFaceColor', 'none', 'BoxEdgeColor', 'k', 'MarkerStyle','+','MarkerColor','r');
hold on;
ylabel({'Pyr mod', 'Electric Field (V/m)'}, 'FontSize', 18);
ylim([0 290]);
set(gca,'FontSize',18, 'LineWidth',2);
for n=1:max(unique(Pyr_4kHz_mod))
    hs = scatter(ones(sum(Pyr_4kHz_mod==n),1) + n-1, Pyr_4kHz_density_mod(Pyr_4kHz_mod == n),"filled",'jitter','on','JitterAmount',0.1);
    hs.MarkerFaceAlpha = 0.5;
    hs.MarkerFaceColor = 'b'
end
box off
saveas(gcf,'SupplFig3_Pyr_4kHz_I_mod_elecF.tif')
saveas(gcf,'SupplFig3_Pyr_4kHz_I_mod_elecF.fig')

% edits for electric field calculation
PV_4kHz_density_sin = PV_4kHz_density_sin*0.33;
fig3 = figure;
boxes = boxchart(PV_4kHz_density_sin)
set(boxes, 'BoxFaceColor', 'none', 'BoxEdgeColor', 'k', 'MarkerStyle','+','MarkerColor','r');
hold on;
ylabel({'PV sin', 'Electric Field (V/m)'}, 'FontSize', 18);
ylim([0 290]);
set(gca,'FontSize',18, 'LineWidth',2);
for n=1:max(unique(PV_4kHz_sin))
    hs = scatter(ones(sum(PV_4kHz_sin==n),1) + n-1, PV_4kHz_density_sin(PV_4kHz_sin == n),"filled",'jitter','on','JitterAmount',0.1);
    hs.MarkerFaceAlpha = 0.5;
    hs.MarkerFaceColor = 'm'
end
box off
saveas(gcf,'SupplFig3_PV_4kHz_I_sin_elecF.tif')
saveas(gcf,'SupplFig3_PV_4kHz_I_sin_elecF.fig')

% edits for electric field calculation
PV_4kHz_density_mod = PV_4kHz_density_mod*0.33;
fig4 = figure;
boxes = boxchart(PV_4kHz_density_mod)
set(boxes, 'BoxFaceColor', 'none', 'BoxEdgeColor', 'k', 'MarkerStyle','+','MarkerColor','r');
hold on;
ylabel({'PV mod', 'Electric Field (V/m)'}, 'FontSize', 18);
ylim([0 290]);
set(gca,'FontSize',18, 'LineWidth',2);
for n=1:max(unique(PV_4kHz_mod))
    hs = scatter(ones(sum(PV_4kHz_mod==n),1) + n-1, PV_4kHz_density_mod(PV_4kHz_mod == n),"filled",'jitter','on','JitterAmount',0.1);
    hs.MarkerFaceAlpha = 0.5;
    hs.MarkerFaceColor = 'm'
end
box off
saveas(gcf,'SupplFig3_PV_4kHz_I_mod_elecF.tif')
saveas(gcf,'SupplFig3_PV_4kHz_I_mod_elecF.fig')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




