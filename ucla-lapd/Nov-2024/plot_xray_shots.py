"""Stacked X-ray traces for a sequence of shots, panels labeled (a)-(e)."""

import os
import sys
from string import ascii_lowercase

import matplotlib.pyplot as plt

sys.path.append(r"C:\Users\hjia9\Documents\GitHub\data-analysis\read")
sys.path.append(r"C:\Users\hjia9\Documents\GitHub\data-analysis")

from read_scope_data import read_trc_data


DATA_DIR = r"E:\good_data\He3kA_B250G500G_pl0t20_uw15t35_P30"
SHOT_NUMBERS = range(22, 26)
FNAME_FMT = "C3--E-ring-p30-z13-x200-xray--{n:05d}.trc"
BDOT_FMT = "C3--E-ring-p30-z13-x200-Bdot--{n:05d}.trc"

EXTRA_TRACE_PATH = r"E:\good_data\He3kA_B250G500G_pl0t20_uw15t35_P30\background\C3--E-ring-p30-z13-x200-xray--00005.trc"

OKABE_ITO = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00',
             '#56B4E9', '#F0E442', '#000000']


def collect_files(data_dir, shot_numbers):
    files = []
    for n in shot_numbers:
        xray_path = os.path.join(data_dir, FNAME_FMT.format(n=n))
        bdot_path = os.path.join(data_dir, BDOT_FMT.format(n=n))
        if os.path.exists(bdot_path) and os.path.exists(xray_path):
            files.append((n, xray_path))
    return files


def configure_style():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 9,
        'axes.labelsize': 10,
        'axes.titlesize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'axes.linewidth': 0.8,
        'lines.linewidth': 0.9,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True,
    })


def plot_shots(files, save_path=None):
    n_files = len(files)
    if n_files == 0:
        raise RuntimeError("No matching shot files found.")

    n_panels = n_files + (1 if EXTRA_TRACE_PATH else 0)

    fig, axes = plt.subplots(
        n_panels, 1, sharex=True,
        figsize=(7.0, 1.6 * n_panels + 0.4),
        constrained_layout=True,
    )
    if n_panels == 1:
        axes = [axes]


    traces = []
    for (shot_num, fpath) in files:
        shot_data, shot_tarr = read_trc_data(fpath)
        traces.append((shot_num, shot_tarr * 1e3, -shot_data))

    t_min = min(t.min() for _, t, _ in traces)
    t_max = max(t.max() for _, t, _ in traces)

    for i, (ax, (shot_num, t_ms, sig)) in enumerate(zip(axes, traces)):
        color = OKABE_ITO[i % len(OKABE_ITO)]
        ax.plot(t_ms, sig, color=color, lw=0.8)

        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.text(0.01, 0.95, f"({ascii_lowercase[i]})",
                transform=ax.transAxes, ha='left', va='top',
                fontweight='bold')

    if EXTRA_TRACE_PATH:
        extra_data, extra_tarr = read_trc_data(EXTRA_TRACE_PATH)
        extra_t_ms = extra_tarr * 1e3
        mask = (extra_t_ms >= t_min) & (extra_t_ms <= t_max)
        ax_extra = axes[-1]
        ax_extra.plot(extra_t_ms[mask], -extra_data[mask],
                      color=OKABE_ITO[n_files % len(OKABE_ITO)],
                      lw=0.8)
        ax_extra.set_yticks([])
        ax_extra.spines['top'].set_visible(False)
        ax_extra.spines['right'].set_visible(False)
        ax_extra.text(0.01, 0.95, f"({ascii_lowercase[n_files]})",
                      transform=ax_extra.transAxes, ha='left', va='top',
                      fontweight='bold')

    axes[0].set_xlim(t_min, t_max)
    axes[-1].set_xlabel("Time (ms)")

    if save_path:
        for ext in ('pdf', 'png'):
            fig.savefig(f"{save_path}.{ext}", dpi=300, bbox_inches='tight')
        print(f"Saved {save_path}.pdf / {save_path}.png")

    plt.show()
    return fig


def main(save=False):
    configure_style()
    files = collect_files(DATA_DIR, SHOT_NUMBERS)
    out_path = None
    if save:
        out_dir = os.path.dirname(os.path.abspath(__file__))
        out_path = os.path.join(out_dir, "xray_shots_22-26")
        plot_shots(files, save_path=out_path)
    else:
        plot_shots(files)


if __name__ == "__main__":

    main(save=False)
