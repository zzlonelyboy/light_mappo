%% Figure 4: Detection Performance by Number of Attacked UAVs (RCS=6, event-likelihood)
clear; close all;

%% --- Load data ---
tbl = readtable('D:\Codes\project_spoofing_detect\light_mappo\isac_spoofing_detection\experiments\neighbor_3way_rcs6_stage2_event_likelihood\by_attack_uav_count_test\attack_count_ablation_summary.csv');

%% --- Parameters ---
n_attack = [1,2,3,4];
fs_names = {'no_neighbor','signed_anomaly_contrast_only','no_bs_residual_signed_only'};
fs_labels = {'F8 (no neighbor)','F9 (+ signed contrast)','F3 (no BS)'};
colors = {[0.4 0.6 0.9],[0.9 0.5 0.3],[0.7 0.3 0.3]};
styles = {'-o','-s','-^'};

figure('Position',[100 100 800 600]);

% --- (a) Event Precision ---
subplot(2,2,1);
for i = 1:3
    idx = strcmp(tbl.feature_set, fs_names{i});
    prec = tbl.event_precision_mean(idx)';
    plot(n_attack, prec, styles{i}, 'Color', colors{i}, 'LineWidth',1.8,...
        'MarkerSize',8, 'MarkerFaceColor',colors{i});
    hold on;
end
ylabel('Event Precision'); xlabel('Number of Attacked UAVs');
title('(a) Event Precision');
ylim([0.75 1.0]); grid on; box on;
legend(fs_labels, 'Location','southwest');

% --- (b) Event Recall ---
subplot(2,2,2);
for i = 1:3
    idx = strcmp(tbl.feature_set, fs_names{i});
    rec = tbl.event_recall_mean(idx)';
    plot(n_attack, rec, styles{i}, 'Color', colors{i}, 'LineWidth',1.8,...
        'MarkerSize',8, 'MarkerFaceColor',colors{i});
    hold on;
end
ylabel('Event Recall'); xlabel('Number of Attacked UAVs');
title('(b) Event Recall');
ylim([0.7 1.0]); grid on; box on;

% --- (c) Event F1 ---
subplot(2,2,3);
for i = 1:3
    idx = strcmp(tbl.feature_set, fs_names{i});
    f1 = tbl.event_f1_mean(idx)';
    plot(n_attack, f1, styles{i}, 'Color', colors{i}, 'LineWidth',1.8,...
        'MarkerSize',8, 'MarkerFaceColor',colors{i});
    hold on;
end
ylabel('Event F1'); xlabel('Number of Attacked UAVs');
title('(c) Event F1');
ylim([0.75 1.0]); grid on; box on;

% --- (d) Mean Delay ---
subplot(2,2,4);
for i = 1:3
    idx = strcmp(tbl.feature_set, fs_names{i});
    delay = tbl.delay_mean_s_mean(idx)';
    plot(n_attack, delay, styles{i}, 'Color', colors{i}, 'LineWidth',1.8,...
        'MarkerSize',8, 'MarkerFaceColor',colors{i});
    hold on;
end
ylabel('Mean Delay (s)'); xlabel('Number of Attacked UAVs');
title('(d) Detection Delay');
grid on; box on;

sgtitle('Figure 4: Performance vs. Attack Scale (RCS=6, K=5, event-likelihood)');
saveas(gcf, 'figures\Figure4_attack_scale.png');
fprintf('Figure 4 saved.\n');
