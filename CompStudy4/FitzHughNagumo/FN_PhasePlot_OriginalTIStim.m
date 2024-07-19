clear all;
freq = [1000;1020]; % Hz
amp = 0.4*[7,-7]; % [35;5];    % No unit
Fs = 18;        % kHz
Twidth = 200;


% Nullclines
% dV/dt = V-V^3/3-W+I
% dW/dt = 0.08(V+0.7-0.8W)
fig=figure('position',[100 100 850 600]);
ax1 = subplot(4,1,1);
hold on;
Vgrid = -5:0.05:5;
W1 = Vgrid-Vgrid.^3/3;
W2 = (Vgrid+1)/0.01;
%vnull = plot(Vgrid, W1, 'r--', 'LineWidth', 1.5);
%wnull = plot(Vgrid, W2, 'k', 'LineWidth', 1.5);
%state = scatter(-1.2, -0.6, 'ro', 'filled');
%xlabel('V_m (Membrane potential)'), ylabel('W');

% [T, V, W] = FitzHughNagumo_Fast(1000, freq, amp, Fs);
[T, S] = FitzHughNagumo_OriginalTIStimOptimizedParameters(Twidth, freq, amp, Fs);
V = S(:, 1);
W = S(:, 2);
input = sum(diag(amp)*cos(2*pi*freq*T'/1000), 1);

ax1 = subplot(3, 1, 1);
hold on;
plot(T, V, 'k', 'LineWidth', 1.5);
%V_state = scatter(T(1), V(1), 'r', 'filled');
xlabel('time (ms)'), ylabel('V_m');

ax2 = subplot(3, 1, 2);
hold on;
plot(T, W, 'k', 'LineWidth', 1.5);
%W_state = scatter(T(1), W(1), 'r', 'filled');
xlabel('time (ms)'), ylabel('W');


ax3 = subplot(3,1,3);
hold on;
plot(T, input, 'k', 'LineWidth', 1.5), xlabel('time (ms)'), ylabel({'Input Current';'Amplitude (au)'});
%input_state = scatter(T(1), input(1), 'r', 'filled');


%
% for i=1:1:length(T)/80     % Change to 1:5:length(T) or even higher number to lower temporal resolution of animation
%     vnull.YData =  Vgrid - Vgrid.^3/3 + input(i);
%     state.XData = V(i);
%     state.YData = W(i);
%     input_state.XData = T(i);
%     input_state.YData = input(i);
%     V_state.XData = T(i);
%     V_state.YData = V(i);
%     W_state.XData = T(i);
%     W_state.YData = W(i);
%     
%     xlim(ax1, [-3.5, 3.5]), ylim(ax1, [-2+min(input), 2+max(input)])
%     xlim(ax2, [min(T), max(T)]), ylim(ax2, [1.1*min(V), 1.1*max(V)])
%     xlim(ax4, [min(T), max(T)]), ylim(ax4, [1.1*min(input), 1.1*max(input)])
%     pause(0.01);        % Change this to control the speed of animation
%     MovieVar(i) = getframe(fig);
% end
% 


%%%%%%%%%%%%
% 
% close all
% [h, w, p] = size(MovieVar(1).cdata);  % use 1st frame to get dimensions
% hf = figure; 
% % resize figure based on frame's w x h, and place at (150, 150)
% set(hf, 'position', [150 150 w h]);
% axis off
% movie(hf,MovieVar);
% %mplay(MovieVar)
% %movie(MovieVar,2)
% myVideo= VideoWriter('myfile.avi');
% open(myVideo);
% for u=1:size(MovieVar,2)
%      frame = MovieVar(u);
%      writeVideo(myVideo, frame);
% end
% % close the writer object
% close(myVideo);
% 
