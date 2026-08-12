% =========================================================================
% Plot: BER vs Eb/N0 (6 codebooks, dashed = impairment-free reference).
% Input:  artifacts/main/sweeps/SCMA_EbN0_Simulation_Results.mat
% Output: artifacts/main/figures/BER_vs_ebn0.{pdf,eps}
% =========================================================================
clear; clc; close all;

script_dir = fileparts(mfilename('fullpath'));
repo_root = fileparts(fileparts(script_dir));
data_file = fullfile(repo_root, 'artifacts', 'main', 'sweeps', 'SCMA_EbN0_Simulation_Results.mat');
figure_dir = fullfile(repo_root, 'artifacts', 'main', 'figures');
if ~exist(figure_dir, 'dir'), mkdir(figure_dir); end

% Load data exported by the Python evaluation script.
if ~exist(data_file, 'file')
    error('Data file not found. Run eval_ber_vs_ebn0.py first.');
end
load(data_file);
if ~exist('BER_95_upper', 'var') || ~exist('BER_is_upper_bound', 'var')
    error('Missing zero-error metadata. Re-run eval_ber_vs_ebn0.py with the revised evaluator.');
end
BER_plot = BER;
BER_plot(BER_is_upper_bound ~= 0) = BER_95_upper(BER_is_upper_bound ~= 0);
EbN0_vec = squeeze(EbN0dB_vec);

% Legend labels (6 codebooks).
labels = {
    'Proposed Codebook', ...
    'DE-Based (Deka et al. [6])', ...
    'Power-Imbalanced (Li et al. [7])', ...
    'Capacity-Based (Zhang et al. [8])', ...
    'Deep Learning (Zheng et al. [11])', ...
    'PN-Resilient (Liu et al. [10])'
};

colors = [
    0.00, 0.45, 0.74;
    0.85, 0.33, 0.10;
    0.93, 0.69, 0.13;
    0.49, 0.18, 0.56;
    0.47, 0.67, 0.19;
    1.00, 0.00, 0.00
];
markers = {'o', 's', '^', 'd', 'x', 'v'};

figure('Position', [100, 100, 600, 480], 'Color', 'w');
hold on;

% BER layout: (n_cond, nCB, n_ebn0) = (2, 6, 13).
%   ie = 1 -> Ideal    (CFO=0, PN=0)        -> dashed, not in legend
%   ie = 2 -> Impaired (CFO=0.03, PN=1e-4)  -> solid + marker, in legend
for i = 1:6
    ber_solid = squeeze(BER_plot(2, i, :));
    ub_solid = squeeze(BER_is_upper_bound(2, i, :)) ~= 0;
    semilogy(EbN0_vec, ber_solid, ['-', markers{i}], 'Color', colors(i,:), ...
        'LineWidth', 2.0, 'MarkerSize', 9, 'MarkerFaceColor', 'none', ...
        'DisplayName', labels{i});

    ber_dash = squeeze(BER_plot(1, i, :));
    ub_dash = squeeze(BER_is_upper_bound(1, i, :)) ~= 0;
    semilogy(EbN0_vec, ber_dash, '--', 'Color', [0.55, 0.55, 0.55], ...
        'LineWidth', 1.2, 'Marker', markers{i}, 'MarkerIndices', 1:2:numel(EbN0_vec), ...
        'MarkerSize', 5, 'MarkerFaceColor', 'none', 'HandleVisibility', 'off');

    % Downward triangles identify zero-error points plotted at the 95% upper bound.
    if any(ub_solid)
        semilogy(EbN0_vec(ub_solid), ber_solid(ub_solid), 'v', 'LineStyle', 'none', ...
            'Color', colors(i,:), 'MarkerFaceColor', 'w', 'HandleVisibility', 'off');
    end
    if any(ub_dash)
        semilogy(EbN0_vec(ub_dash), ber_dash(ub_dash), 'v', 'LineStyle', 'none', ...
            'Color', colors(i,:), 'MarkerFaceColor', 'w', 'HandleVisibility', 'off');
    end
end
semilogy(nan, nan, 'kv', 'LineStyle', 'none', 'MarkerFaceColor', 'w', ...
    'DisplayName', 'Zero-error 95% upper bound');

set(gca, 'YScale', 'log');
grid on;
set(gca, 'XMinorGrid', 'on', 'YMinorGrid', 'on', 'MinorGridLineStyle', ':');
set(gca, 'GridLineStyle', '-', 'GridAlpha', 0.4);
set(gca, 'TickDir', 'in');
set(gca, 'FontName', 'Times New Roman', 'FontSize', 13);
xlabel('E_b/N_0 (dB)', 'FontSize', 14);
ylabel('BER', 'FontSize', 14);
ylim([1e-7, 1e-2]);
xlim([min(EbN0_vec), max(EbN0_vec)]);

% Fig. 2(a) carries the shared legend for all three panels.
box on;
set(gca, 'LineWidth', 1.0);

% Export vector figures.
set(gcf, 'PaperPositionMode', 'auto');
try
    exportgraphics(gcf, fullfile(figure_dir, 'BER_vs_ebn0.eps'), 'ContentType', 'vector', 'BackgroundColor', 'none');
    exportgraphics(gcf, fullfile(figure_dir, 'BER_vs_ebn0.pdf'), 'ContentType', 'vector', 'BackgroundColor', 'none');
    disp('Saved: BER_vs_ebn0.{eps,pdf}');
catch
    fig_pos = get(gcf, 'PaperPosition');
    set(gcf, 'PaperSize', [fig_pos(3) fig_pos(4)]);
    print(gcf, '-depsc2', '-r600', '-loose', fullfile(figure_dir, 'BER_vs_ebn0.eps'));
end
