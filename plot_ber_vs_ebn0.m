% =========================================================================
% Plot: BER vs Eb/N0 (6 codebooks, dashed = impairment-free reference).
% Input:  SCMA_EbN0_Simulation_Results.mat (from eval_ber_vs_ebn0.py)
% Output: BER_vs_ebn0.{pdf,eps}
% =========================================================================
clear; clc; close all;

% Load data exported by the Python evaluation script.
load('SCMA_EbN0_Simulation_Results.mat');
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
    'Capacitybased (Zhang et al. [8])', ...
    'Deep Learning (Zheng et al. [11])', ...
    'PN-Resilient (Liu et al. [10])'
};

colors  = lines(6);
markers = {'o', 's', '^', 'd', 'x', '*'};

figure('Position', [100, 100, 600, 480], 'Color', 'w');
hold on;

% BER layout: (n_cond, nCB, n_ebn0) = (2, 6, 13).
%   ie = 1 -> Ideal    (CFO=0, PN=0)        -> dashed, not in legend
%   ie = 2 -> Impaired (CFO=0.03, PN=1e-4)  -> solid + marker, in legend
for i = 1:6
    ber_solid = squeeze(BER_plot(2, i, :));
    ub_solid = squeeze(BER_is_upper_bound(2, i, :)) ~= 0;
    semilogy(EbN0_vec, ber_solid, ['-', markers{i}], 'Color', colors(i,:), ...
        'LineWidth', 1.5, 'MarkerSize', 8, 'MarkerFaceColor', 'none', ...
        'DisplayName', labels{i});

    ber_dash = squeeze(BER_plot(1, i, :));
    ub_dash = squeeze(BER_is_upper_bound(1, i, :)) ~= 0;
    semilogy(EbN0_vec, ber_dash, '--', 'Color', colors(i,:), ...
        'LineWidth', 1.0, 'HandleVisibility', 'off');

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
set(gca, 'FontName', 'Times New Roman', 'FontSize', 12);
xlabel('E_b/N_0 (dB)');
ylabel('BER');
ylim([1e-7, 1e-2]);
xlim([min(EbN0_vec), max(EbN0_vec)]);

lgd = legend('Location', 'southwest');
lgd.ItemTokenSize = [30, 18];
box on;
set(gca, 'LineWidth', 1.0);

% Export vector figures.
set(gcf, 'PaperPositionMode', 'auto');
try
    exportgraphics(gcf, 'BER_vs_ebn0.eps', 'ContentType', 'vector', 'BackgroundColor', 'none');
    exportgraphics(gcf, 'BER_vs_ebn0.pdf', 'ContentType', 'vector', 'BackgroundColor', 'none');
    disp('Saved: BER_vs_ebn0.{eps,pdf}');
catch
    fig_pos = get(gcf, 'PaperPosition');
    set(gcf, 'PaperSize', [fig_pos(3) fig_pos(4)]);
    print(gcf, '-depsc2', '-r600', '-loose', 'BER_vs_ebn0.eps');
end
