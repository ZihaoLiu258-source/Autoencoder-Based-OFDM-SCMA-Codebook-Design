% =========================================================================
% Plot: BER vs phase-noise sigma (6 codebooks, dashed = ideal CFO).
% Input:  SCMA_CFO_Simulation_Results.mat (from eval_ber_vs_phasenoise.py)
% Output: BER_vs_phasenoise.{pdf,eps}
% =========================================================================
clear; clc; close all;

% Load data exported by the Python evaluation script.
if ~exist('SCMA_CFO_Simulation_Results.mat', 'file')
    error('Data file not found. Run eval_ber_vs_phasenoise.py first.');
end
load('SCMA_CFO_Simulation_Results.mat');
if ~exist('BER_95_upper', 'var') || ~exist('BER_is_upper_bound', 'var')
    error('Missing zero-error metadata. Re-run eval_ber_vs_phasenoise.py with the revised evaluator.');
end
BER_plot = BER_3D;
BER_plot(BER_is_upper_bound ~= 0) = BER_95_upper(BER_is_upper_bound ~= 0);

% Ensure the x-axis vector is a column.
sigma_vec = squeeze(sigma_vec);
sigma_vec = sigma_vec(:);

% Legend labels (6 codebooks).
labels = {
    'Proposed Codebook', ...
    'DE-Based (Deka et al. [6])', ...
    'Power-Imbalanced (Li et al. [7])', ...
    'Capacity-Based (Zhang et al. [8])', ...
    'Deep Learning (Zheng et al. [11])', ...
    'PN-Resilient (Liu et al. [10])'
};
% Colors and markers. The 6th codebook (PN-Resilient) is drawn in red.
colors = [
    0.00, 0.45, 0.74;  % blue
    0.85, 0.33, 0.10;  % orange
    0.93, 0.69, 0.13;  % yellow
    0.49, 0.18, 0.56;  % purple
    0.47, 0.67, 0.19;  % green
    1.00, 0.00, 0.00   % red (PN-Resilient)
];
markers = {'o', 's', '^', 'd', 'x', 'v'};

figure('Position', [100, 100, 650, 520], 'Color', 'w');
hold on;

nCB = size(BER_3D, 3);
for i = 1:nCB
    % BER_3D layout: (eps_vec, sigma_vec, nCB). Two CFO points per codebook.
    ber_eps_0  = squeeze(BER_plot(1, :, i));  % eps = 0.0
    ber_eps_04 = squeeze(BER_plot(2, :, i));  % eps = 0.04
    ub_eps_0   = squeeze(BER_is_upper_bound(1, :, i)) ~= 0;
    ub_eps_04  = squeeze(BER_is_upper_bound(2, :, i)) ~= 0;

    ber_eps_0  = ber_eps_0(:);
    ber_eps_04 = ber_eps_04(:);
    ub_eps_0   = ub_eps_0(:);
    ub_eps_04  = ub_eps_04(:);

    % Light dashed line with the matching marker: ideal CFO (eps = 0).
    semilogy(sigma_vec, ber_eps_0, '--', 'Color', [0.55, 0.55, 0.55], ...
        'LineWidth', 1.2, 'Marker', markers{i}, 'MarkerIndices', 1:2:numel(sigma_vec), ...
        'MarkerSize', 5, 'MarkerFaceColor', 'none', 'HandleVisibility', 'off');

    % Solid line: severe CFO (eps = 0.04). PN-Resilient drawn slightly thicker.
    lw = 2.0;
    semilogy(sigma_vec, ber_eps_04, ['-', markers{i}], 'Color', colors(i,:), ...
        'LineWidth', lw, 'MarkerSize', 9, 'MarkerFaceColor', 'none', ...
        'DisplayName', labels{i});

    % Downward triangles identify zero-error points plotted at the 95% upper bound.
    if any(ub_eps_0)
        semilogy(sigma_vec(ub_eps_0), ber_eps_0(ub_eps_0), 'v', 'LineStyle', 'none', ...
            'Color', colors(i,:), 'MarkerFaceColor', 'w', 'HandleVisibility', 'off');
    end
    if any(ub_eps_04)
        semilogy(sigma_vec(ub_eps_04), ber_eps_04(ub_eps_04), 'v', 'LineStyle', 'none', ...
            'Color', colors(i,:), 'MarkerFaceColor', 'w', 'HandleVisibility', 'off');
    end
end
semilogy(nan, nan, 'kv', 'LineStyle', 'none', 'MarkerFaceColor', 'w', ...
    'DisplayName', 'Zero-error 95% upper bound');

set(gca, 'YScale', 'log');
grid on;
set(gca, 'XMinorGrid', 'on', 'YMinorGrid', 'on', 'MinorGridLineStyle', ':');
set(gca, 'GridLineStyle', '-', 'GridAlpha', 0.3);
set(gca, 'TickDir', 'in');
set(gca, 'FontName', 'Times New Roman', 'FontSize', 13);
xlabel('Phase Noise \sigma (rad)', 'FontSize', 14);
ylabel('Bit Error Rate (BER)', 'FontSize', 14);
ylim([1e-6, 1e-1]);
xlim([0, max(sigma_vec)]);

% Fig. 2(a) carries the shared legend for all three panels.
box on;
set(gca, 'LineWidth', 1.2);

% Export vector figures.
set(gcf, 'PaperPositionMode', 'auto');
try
    exportgraphics(gcf, 'BER_vs_phasenoise.pdf', 'ContentType', 'vector');
    exportgraphics(gcf, 'BER_vs_phasenoise.eps', 'ContentType', 'vector');
    disp('Saved: BER_vs_phasenoise.{pdf,eps}');
catch
    saveas(gcf, 'BER_vs_phasenoise.png');
    disp('exportgraphics unavailable; saved BER_vs_phasenoise.png instead.');
end
