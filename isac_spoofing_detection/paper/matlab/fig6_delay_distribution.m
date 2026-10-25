%% Figure 6: Cumulative Detection Delay Distribution
% 对比 BCE vs Event-Likelihood, RCS=6 vs RCS=3
clear; close all;

%% --- Data: detection_rate_within_s from evaluation_test.json ---
% Values extracted manually from test evaluations (F9 feature set)
% Format: [0.5s, 1.0s, 2.0s, 5.0s] thresholds
thresholds = [0.5, 1.0, 2.0, 5.0];

% RCS=6 BCE (F9) — test split
rcs6_bce = [0.2817, 0.4349, 0.5761, 0.8024];
% RCS=6 Event-Likelihood (F9) — test split
rcs6_evl = [0.2853, 0.4323, 0.5735, 0.7908];
% RCS=3 BCE (F9) — test split
rcs3_bce = [0.2140, 0.3697, 0.4956, 0.7329];
% RCS=3 Event-Likelihood (F9) — test split
rcs3_evl = [0.2165, 0.3723, 0.4927, 0.7198];

figure('Position',[100 100 900 380]);

subplot(1,2,1);
plot(thresholds, rcs6_bce*100, 'b-o', 'LineWidth',1.8, 'MarkerSize',8, 'MarkerFaceColor','b');
hold on;
plot(thresholds, rcs6_evl*100, 'r-s', 'LineWidth',1.8, 'MarkerSize',8, 'MarkerFaceColor','r');
xlabel('Time from Attack Onset (s)');
ylabel('Cumulative Detection Rate (%)');
title('(a) RCS = 6');
legend('BCE (step-level)', 'Event-Likelihood', 'Location','southeast');
xlim([0 5.5]); ylim([0 90]);
grid on; box on;

subplot(1,2,2);
plot(thresholds, rcs3_bce*100, 'b-o', 'LineWidth',1.8, 'MarkerSize',8, 'MarkerFaceColor','b');
hold on;
plot(thresholds, rcs3_evl*100, 'r-s', 'LineWidth',1.8, 'MarkerSize',8, 'MarkerFaceColor','r');
xlabel('Time from Attack Onset (s)');
ylabel('Cumulative Detection Rate (%)');
title('(b) RCS = 3');
legend('BCE (step-level)', 'Event-Likelihood', 'Location','southeast');
xlim([0 5.5]); ylim([0 90]);
grid on; box on;

sgtitle('Figure 6: Cumulative Detection Delay Distribution (F9 feature set)');
saveas(gcf, 'figures\Figure6_delay_distribution.png');
fprintf('Figure 6 saved.\n');
