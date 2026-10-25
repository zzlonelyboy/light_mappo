%% Figure 3: Neighbor Gain by Self-Confidence Stratification
% 双Y轴图: 左侧事件召回率, 右侧邻居净增益
clear; close all;

%% --- Load RCS=6 data ---
rcs6 = readtable('..\experiments\neighbor_3way_rcs6_stage2_event_likelihood\neighbor_gain_by_self_confidence_test\event_evidence_summary.csv');
rcs3 = readtable('..\experiments\neighbor_3way_rcs3_stage2_event_likelihood\neighbor_gain_by_self_confidence_test\event_evidence_summary.csv');

%% --- Data preparation ---
% Bin labels and x-tick positions
bins = categorical({'Very Low\n[0,0.3)','Ambiguous\n[0.3,0.5)','Boundary\n[0.5,0.7)','Strong\n[0.7,0.9)','Very Strong\n[0.9,1]'});
x = 1:5;

% RCS=6
r6_ref_recall  = rcs6.reference_event_recall_mean';
r6_comp_recall = rcs6.comparison_event_recall_mean';
r6_net_rate    = rcs6.net_recovery_rate_mean';
r6_net_std     = rcs6.net_recovery_rate_std';

% RCS=3
r3_ref_recall  = rcs3.reference_event_recall_mean';
r3_comp_recall = rcs3.comparison_event_recall_mean';
r3_net_rate    = rcs3.net_recovery_rate_mean';
r3_net_std     = rcs3.net_recovery_rate_std';

%% --- Plot ---
figure('Position',[100 100 900 420]);

% === Subplot (a): RCS = 6 ===
subplot(1,2,1);
yyaxis left;
b1 = bar(x, [r6_ref_recall; r6_comp_recall]', 'grouped');
b1(1).FaceColor = [0.4 0.6 0.9];  b1(1).DisplayName = 'F8 (no neighbor)';
b1(2).FaceColor = [0.9 0.5 0.3];  b1(2).DisplayName = 'F9 (+ signed contrast)';
ylabel('Event Recall');
ylim([0 1.05]);
set(gca,'XTick',x,'XTickLabel',{'Very Low','Ambiguous','Boundary','Strong','Very Strong'});

yyaxis right;
errorbar(x, r6_net_rate, r6_net_std, 'k-o','LineWidth',1.5,'MarkerSize',8,'MarkerFaceColor','k');
ylabel('Net Recovery Rate \DeltaRecall');
ylim([-0.6 0.6]);
hline = refline(0,0); hline.Color = [0.5 0.5 0.5]; hline.LineStyle = '--';
title('(a) RCS = 6');
legend([b1; hline],'F8 (no neighbor)','F9 (+ contrast)','Zero-gain','Location','northwest');
grid on; box on;

% === Subplot (b): RCS = 3 ===
subplot(1,2,2);
yyaxis left;
b2 = bar(x, [r3_ref_recall; r3_comp_recall]', 'grouped');
b2(1).FaceColor = [0.4 0.6 0.9];  b2(1).DisplayName = 'F8 (no neighbor)';
b2(2).FaceColor = [0.9 0.5 0.3];  b2(2).DisplayName = 'F9 (+ signed contrast)';
ylabel('Event Recall');
ylim([0 1.05]);
set(gca,'XTick',x,'XTickLabel',{'Very Low','Ambiguous','Boundary','Strong','Very Strong'});

yyaxis right;
errorbar(x, r3_net_rate, r3_net_std, 'k-o','LineWidth',1.5,'MarkerSize',8,'MarkerFaceColor','k');
ylabel('Net Recovery Rate \DeltaRecall');
ylim([-0.6 0.6]);
hline2 = refline(0,0); hline2.Color = [0.5 0.5 0.5]; hline2.LineStyle = '--';
title('(b) RCS = 3');
legend([b2; hline2],'F8 (no neighbor)','F9 (+ contrast)','Zero-gain','Location','northwest');
grid on; box on;

sgtitle('Figure 3: Neighbor Gain Stratified by Self-Detector Confidence');

saveas(gcf, 'figures\Figure3_neighbor_gain_confidence.png');
fprintf('Figure 3 saved.\n');
