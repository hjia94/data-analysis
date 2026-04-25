"""X-ray traces in two side-by-side panels (a) and (b) with pulse detection."""

import os
import re
import sys

import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_HERE)
sys.path.append(r"C:\Users\hjia9\Documents\GitHub\data-analysis\read")
sys.path.append(r"C:\Users\hjia9\Documents\GitHub\data-analysis")

from read_scope_data import read_trc_data
from process_xray_bdot import process_shot_xray


PATH_A = r"E:\good_data\He3kA_B250G500G_pl0t20_uw15t35_P30\background\C3--E-ring-p30-z13-x200-xray--00003.trc"
PATH_B = r"E:\good_data\He3kA_B250G500G_pl0t20_uw15t35_P30\C3--E-ring-p30-z13-x200-xray--00022.trc"

OKABE_ITO = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00',
             '#56B4E9', '#F0E442', '#000000']


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


def _file_number(path):
    """Extract zero-padded shot number from filename, e.g. '00022' from '...--00022.trc'."""
    m = re.search(r'--(\d{5})\.trc$', os.path.basename(path))
    if not m:
        raise ValueError(f"Cannot extract shot number from: {path}")
    return m.group(1)


def plot_panels(save_path=None):
    fig, axes = plt.subplots(
        1, 2,
        figsize=(7.0, 3.0),
        constrained_layout=True,
    )

    # Calculate y-axis limit based on both datasets
    max_vals = []
    for path in [PATH_A, PATH_B]:
        if path:
            data, tarr = read_trc_data(path)
            max_vals.append(max(-data))
    
    y_max = max(max_vals) * 1.2
    
    for ax, path, label, color in zip(axes, [PATH_A, PATH_B], ['(a)', '(b)'], OKABE_ITO):
        if path:
            data, tarr = read_trc_data(path)
            t_ms = tarr * 1e3

            if path == PATH_A:
                max_sample = int(0.4 * len(t_ms))
                ax.plot(t_ms[:max_sample], -data[:max_sample], color=color, lw=0.8)
            if path == PATH_B:
                ax.plot(t_ms, -data, color=color, lw=0.8)
        
        ax.set_ylim(0, y_max)

            # pulse_times, pulse_amps = process_shot_xray(_file_number(path), os.path.dirname(path))
            # ax.scatter(pulse_times, pulse_amps, s=10, color='#CC79A7', alpha=0.8, zorder=3)

        ax.set_yticks([])
        if ax == axes[0]:
            ax.set_ylabel("X-ray signal (a.u.)")
        ax.set_xlabel("Time (ms)")

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.text(0.01, 0.95, label, transform=ax.transAxes, ha='left', va='top', fontweight='bold')

    if save_path:
        for ext in ('pdf', 'png'):
            fig.savefig(f"{save_path}.{ext}", dpi=300, bbox_inches='tight')
        print(f"Saved {save_path}.pdf / {save_path}.png")

    plt.show()
    return fig


def main(save=False):
    configure_style()
    out_path = None
    if save:
        out_dir = os.path.dirname(os.path.abspath(__file__))
        out_path = os.path.join(out_dir, "xray_two_panels")
    plot_panels(save_path=out_path)


if __name__ == "__main__":
    main(save=False)
