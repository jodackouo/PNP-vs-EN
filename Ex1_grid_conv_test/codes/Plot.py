"""
@author: Josaphat Dackouo

This code generates the plots for the grid convergence test.
RESULTS_ON_FILES : contains files for PNP
RESULTS_OFF_FILES : contains files for EN.

Figures are saved in pdf and png formats if SAVE_FIGURES = True in the figures folder

NB: Do not forget to adapt the file name and the domain length in the table reader function.
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# ----------------------------
# User options
# ----------------------------
RESULTS_ON_FILES = [
    r"../file/results_dx10_bcOpnpON.txt",
    r"../file/results_dx9_bcOpnpON.txt",
    r"../file/results_dx8_bcOpnpON.txt",
    ]

RESULTS_OFF_FILES = [
    r"../file/results_dx10_bcOpnpOFF.txt",
    r"../file/results_dx9_bcOpnpOFF.txt",
    r"../file/results_dx8_bcOpnpOFF.txt",
    ]

SAVE_FIGURES = False
OUTDIR = "figures"
SHOW_PLOTS = True
SEP_BETWEEN_GROUPS = "||"
SAVE_FORMATS = ("pdf", "png")

# ----------------------------
# Readability scaling for article
# ----------------------------
TARGET_REDUCTION = 1.0 / 3.0
MAG = max(1.0, 1.0 / TARGET_REDUCTION)

BASE_DPI = 300

BASE_FONTS = {
    "axes_label": 5.0,
    "axes_title": 5.5,
    "legend": 4.5,
    "xtick": 5.0,
    "ytick": 5.0,
    "colorbar_label": 5.0,
    "colorbar_ticks": 5.0,
}

BASE_LINEWIDTH = 0.9
BASE_MARKERSIZE = 4.0

# ----------------------------
# Consistent colors for each dx
# ----------------------------
DX_COLORS = {
    1e-8: "#1f77b4",   # Blue
    1e-9: "#ff7f0e",   # Orange
    1e-10: "#2ca02c",  # Green
}

line_styles = ["-", "--", "-."]
markers = ["o", "s", "^"]

def get_dx_color(dx):
    """Return the same color for a given dx in every subplot."""
    for dx_ref, color in DX_COLORS.items():
        if np.isclose(dx, dx_ref, rtol=1e-6, atol=0.0):
            return color
    raise ValueError(f"No color defined for dx={dx}.")

def get_dx_style(dx):
    """Return a consistent line style and marker for each dx."""
    dx_values = list(DX_COLORS.keys())
    for i, dx_ref in enumerate(dx_values):
        if np.isclose(dx, dx_ref, rtol=1e-6, atol=0.0):
            return line_styles[i % len(line_styles)], markers[i % len(markers)]
    raise ValueError(f"No line style defined for dx={dx}")

def set_pub_style(mag=MAG):
    """Configure matplotlib rcParams for publication-ready figures."""
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "sans serif",
        "font.serif": ["Helvetica"],
        "axes.labelsize": BASE_FONTS["axes_label"] * mag,
        "axes.titlesize": BASE_FONTS["axes_title"] * mag,
        "legend.fontsize": BASE_FONTS["legend"] * mag,
        "xtick.labelsize": BASE_FONTS["xtick"] * mag,
        "ytick.labelsize": BASE_FONTS["ytick"] * mag,
        "figure.dpi": int(BASE_DPI * mag),
        "savefig.dpi": int(BASE_DPI * mag),
        "lines.linewidth": BASE_LINEWIDTH * mag,
        "axes.linewidth": 0.7 * mag,
        "mathtext.fontset": "cm",
    })

# ----------------------------
# Table reader
# ----------------------------
def read_pretty_table(filename, sep_between_groups=SEP_BETWEEN_GROUPS):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")

    with open(filename, "r") as f:
        lines = [ln.rstrip("\n") for ln in f.readlines()]

    if len(lines) < 2:
        raise ValueError("Table file too short / unexpected format.")

    header = lines[0]
    bc_match = re.search(r"BCs=([A-Za-z0-9_\-]+)", header)
    BCs = bc_match.group(1) if bc_match else "unknown"

    dx_matches = re.findall(r"dx=([0-9.eE+\-]+)", header)
    if len(dx_matches) == 0:
        raise ValueError("Could not parse dx values from header line.")

    dx_values = np.array([float(d) for d in dx_matches])
    M = len(dx_values)

    start_idx = 2
    while start_idx < len(lines) and (set(lines[start_idx].strip()) <= set("- ") or lines[start_idx].strip() == ""):
        start_idx += 1

    data_lines = [ln for ln in lines[start_idx:] if ln.strip() != "" and not set(ln.strip()) <= set("- ")]
    if len(data_lines) == 0:
        raise ValueError("No data rows found in table.")

    sep_token = f" {sep_between_groups} "
    rows = []

    for ln in data_lines:
        parts = ln.strip().split(None, 1)
        if len(parts) < 2:
            continue
        _, rest = parts
        groups = [g.strip() for g in rest.split(sep_token)]
        if len(groups) != M:
            groups = [g.strip() for g in rest.split(sep_between_groups)]
        if len(groups) != M:
            raise ValueError(f"Number of groups does not match dx count in line:\n{ln}")

        row_vals = []
        for g in groups:
            tokens = g.split()
            if len(tokens) == 0:
                row_vals.append((np.nan, np.nan, np.nan, np.nan))
            elif len(tokens) == 3:
                row_vals.append((float(tokens[0]), float(tokens[1]), float(tokens[2]), 0.0))
            elif len(tokens) >= 4:
                row_vals.append((float(tokens[0]), float(tokens[1]), float(tokens[2]), float(tokens[3])))
            else:
                row_vals.append((np.nan, np.nan, np.nan, np.nan))
        rows.append(row_vals)

    Nx = len(rows)
    Na_mat = np.zeros((M, Nx))
    Cl_mat = np.zeros((M, Nx))
    V_mat = np.zeros((M, Nx))
    PH_mat = np.zeros((M, Nx))

    for i_row, row_vals in enumerate(rows):
        for j_group, (na, cl, v, ph) in enumerate(row_vals):
            Na_mat[j_group, i_row] = na
            Cl_mat[j_group, i_row] = cl
            V_mat[j_group, i_row] = v
            PH_mat[j_group, i_row] = ph

    L_domain = 1e-6
    x = np.linspace(0.0, L_domain, Nx)
    for i in range(Nx):
        x[i] = (-dx_values / 2.0) + i * dx_values

    return Na_mat, Cl_mat, V_mat, PH_mat, dx_values, x, BCs

def format_dx_to_latex_exp(dx_value):
    exponent = int(np.round(np.log10(dx_value)))
    return fr"$\Delta x=10^{{{exponent}}}$ m"

# ----------------------------
# Combined plot function
# ----------------------------
def plot_unified_composite(on_files, off_files, filename_base="unified_composite", save=SAVE_FIGURES):
    set_pub_style()

    fig, axes = plt.subplots(2, 3, figsize=(17, 10), sharex='col')
    subplot_labels = [["(a)", "(b)", "(c)"], ["(d)", "(e)", "(f)"]]
    BCs = "unknown"

    # ========================================================
    # ROW 1: ELECTRONEUTRALITY (EN) [OFF]
    # ========================================================
    ax_c1, ax_p1, ax_z1 = axes[0, 0], axes[0, 1], axes[0, 2]

    for ax in [ax_c1, ax_p1, ax_z1]:
        ax.tick_params(labelbottom=True)

    # ========================================================
    # a) Concentration EN
    # ========================================================
    axins_a = inset_axes(ax_c1, width="50%", height="50%", loc="center right", borderpad=0.9)
    na_handles_a, cl_handles_a = [], []
    path_b = off_files[0] if off_files else None

    if path_b and os.path.exists(path_b):
        Na_en, Cl_en, V_en, _, dx_vals, x_phys, bc = read_pretty_table(path_b)
        BCs = bc
        x_micro = x_phys * 1e6

        for i, dx in enumerate(dx_vals):
            mask = ~np.isnan(Na_en[i, :])
            color = get_dx_color(dx)
            ls, marker = get_dx_style(dx)

            ax_c1.plot(x_micro[mask], Na_en[i, mask], color=color, lw=3.8, ls=ls, marker=marker, markevery=max(1, len(x_micro[mask]) // 10), markersize=3.0)
            ax_c1.plot(x_micro[mask], Cl_en[i, mask], color=color, lw=3.8, ls=":")

            axins_a.plot(x_micro[mask], Na_en[i, mask], color=color, lw=3.8, ls=ls)
            axins_a.plot(x_micro[mask], Cl_en[i, mask], color=color, lw=3.8, ls=":")

            exponent = int(np.round(np.log10(dx)))
            na_handles_a.append(Line2D([0], [0], color=color, lw=3.8, ls=ls, marker=marker, markersize=3.0, label=rf"$\mathrm{{Na}}^+ \ (\Delta x=10^{{{exponent}}}\,\mathrm{{m}})$"))
            cl_handles_a.append(Line2D([0], [0], color=color, lw=3.8, ls=":", label=rf"$\mathrm{{Cl}}^- \ (\Delta x=10^{{{exponent}}}\,\mathrm{{m}})$"))

    ax_c1.set_ylabel(r"$C_{\mathrm{EN}}$ ($\mathrm{mol\,m}^{-3}$)", fontsize=BASE_FONTS["axes_label"] * MAG * 2.2)

    leg_na_a = ax_c1.legend(handles=na_handles_a, fontsize=21.5, frameon=False, loc="upper center", ncol=3, columnspacing=0.6, handletextpad=0.2)
    ax_c1.add_artist(leg_na_a)
    ax_c1.legend(handles=cl_handles_a, fontsize=21.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.90), ncol=3, columnspacing=0.6, handletextpad=0.2)

    ax_c1.tick_params(axis="both", which="major", length=2*MAG, width=1*MAG)
    ax_c1.tick_params(axis="both", which="minor", length=1*MAG, width=0.3*MAG)
    ax_c1.tick_params(axis="both", labelsize=BASE_FONTS["ytick"] * MAG * 1.4)

    axins_a.set_xlim(0.1, 0.2)
    axins_a.set_ylim(0, 0.2)
    axins_a.set_xticks([0.10, 0.15, 0.20])
    axins_a.set_yticks([0.0, 0.1, 0.2])
    axins_a.tick_params(labelsize=11.0)
    mark_inset(ax_c1, axins_a, loc1=2, loc2=4, fc="none", ec="0.4", ls="--", lw=2.0)

    # ========================================================
    # b) Potential EN
    # ========================================================
    d_handles = []
    for j, path in enumerate(off_files):
        if not os.path.exists(path):
            continue
        _, _, V_en, _, dx_vals, x_phys, _ = read_pretty_table(path)
        x_micro = x_phys * 1e6
        for i, dx in enumerate(dx_vals):
            mask = ~np.isnan(V_en[i, :])
            color = get_dx_color(dx)
            ls, marker = get_dx_style(dx)
            label_str = format_dx_to_latex_exp(dx)
            scatter_plot = ax_p1.scatter(x_micro[mask], V_en[i, mask], color=color, label=label_str)
            d_handles.append((dx, scatter_plot, label_str))

    ax_p1.tick_params(axis="both", which="major", length=2*MAG, width=1*MAG)
    ax_p1.tick_params(axis="both", which="minor", length=1*MAG, width=0.3*MAG)
    ax_p1.tick_params(axis="both", labelsize=BASE_FONTS["ytick"] * MAG * 1.4)

    d_handles.sort(key=lambda item: item[0])
    ax_p1.set_ylim(-0.06, 0)
    ax_p1.set_ylabel(r"$\Psi_{\mathrm{EN}}$ ($\mathrm{V}$)", fontsize=BASE_FONTS["axes_label"] * MAG * 2.2)
    ax_p1.legend(handles=[item[1] for item in d_handles], labels=[item[2] for item in d_handles], fontsize=21.5, frameon=False, loc="best")

    # ========================================================
    # c) Zoom Potential EN
    # ========================================================
    zoom_handles_c = []
    for j, path in enumerate(off_files):
        if not os.path.exists(path):
            continue
        _, _, V_en, _, dx_vals, x_phys, _ = read_pretty_table(path)
        x_micro = x_phys * 1e6
        for i, dx in enumerate(dx_vals):
            mask = ~np.isnan(V_en[i, :])
            color = get_dx_color(dx)
            ls, marker = get_dx_style(dx)
            label_str = format_dx_to_latex_exp(dx)
            scatter_plot = ax_z1.scatter(x_micro[mask], V_en[i, mask], color=color, lw=3.8, ls=ls, label=label_str)
            zoom_handles_c.append((dx, scatter_plot, label_str))

    ax_z1.tick_params(axis="both", which="major", length=2*MAG, width=1*MAG)
    ax_z1.tick_params(axis="both", which="minor", length=1*MAG, width=0.3*MAG)
    ax_z1.tick_params(axis="both", labelsize=BASE_FONTS["ytick"] * MAG * 1.4)

    zoom_handles_c.sort(key=lambda item: item[0])
    ax_z1.set_xlim(0.1, 0.2)
    ax_z1.set_ylim(-0.03, -0.01)
    ax_z1.set_xticks([0.10, 0.15, 0.20])
    ax_z1.set_yticks([-0.03, -0.02, -0.01])
    ax_z1.set_ylabel(r"$\Psi_{\mathrm{EN}}$ ($\mathrm{V}$)", fontsize=BASE_FONTS["axes_label"] * MAG * 2.2)
    ax_z1.legend(handles=[item[1] for item in zoom_handles_c], labels=[item[2] for item in zoom_handles_c], fontsize=21.5, frameon=False, loc="best")

    # ========================================================
    # ROW 2: POISSON-NERNST-PLANCK (PNP) [ON]
    # ========================================================
    ax_c2, ax_p2, ax_z2 = axes[1, 0], axes[1, 1], axes[1, 2]

    # ========================================================
    # d) Concentration PNP
    # ========================================================
    axins_d = inset_axes(ax_c2, width="50%", height="50%", loc="center right", borderpad=0.9)
    na_handles_d, cl_handles_d = [], []
    path_a = on_files[0] if on_files else None

    if path_a and os.path.exists(path_a):
        Na, Cl, _, _, dx_vals, x_phys, _ = read_pretty_table(path_a)
        x_micro = x_phys * 1e6

        for i, dx in enumerate(dx_vals):
            mask = ~np.isnan(Na[i, :])
            color = get_dx_color(dx)
            ls, marker = get_dx_style(dx)

            ax_c2.plot(x_micro[mask], Na[i, mask], color=color, lw=3.8, ls=ls, marker=marker, markevery=max(1, len(x_micro[mask]) // 10), markersize=3.0)
            ax_c2.plot(x_micro[mask], Cl[i, mask], color=color, lw=3.8, ls=":")

            axins_d.plot(x_micro[mask], Na[i, mask], color=color, lw=3.8, ls=ls, marker=marker, markevery=max(1, len(x_micro[mask]) // 10), markersize=3.0)
            axins_d.plot(x_micro[mask], Cl[i, mask], color=color, lw=3.8, ls=":")

            exponent = int(np.round(np.log10(dx)))
            na_handles_d.append(Line2D([0], [0], color=color, lw=3.8, ls=ls, marker=marker, markersize=3.0, label=rf"$\mathrm{{Na}}^+ \ (\Delta x=10^{{{exponent}}}\,\mathrm{{m}})$"))
            cl_handles_d.append(Line2D([0], [0], color=color, lw=3.8, ls=":", label=rf"$\mathrm{{Cl}}^- \ (\Delta x=10^{{{exponent}}}\,\mathrm{{m}})$"))

    ax_c2.set_ylabel(r"$C_{\mathrm{PNP}}$ ($\mathrm{mol\,m}^{-3}$)", fontsize=BASE_FONTS["axes_label"] * MAG * 2.2)

    ax_c2.tick_params(axis="both", which="major", length=2*MAG, width=1*MAG)
    ax_c2.tick_params(axis="both", which="minor", length=1*MAG, width=0.3*MAG)
    ax_c2.tick_params(axis="both", labelsize=BASE_FONTS["ytick"] * MAG * 1.4)

    axins_d.set_xlim(0.1, 0.2)
    axins_d.set_ylim(0, 0.2)
    axins_d.set_xticks([0.10, 0.15, 0.20])
    axins_d.set_yticks([0.0, 0.1, 0.2])
    axins_d.tick_params(labelsize=11.0)
    mark_inset(ax_c2, axins_d, loc1=2, loc2=4, fc="none", ec="0.4", ls="--", lw=2.0)

    # ========================================================
    # e) Potential PNP
    # ========================================================
    c_handles = []
    for j, path in enumerate(on_files):
        if not os.path.exists(path):
            continue
        _, _, V_pnp, _, dx_vals, x_phys, _ = read_pretty_table(path)
        x_micro = x_phys * 1e6
        for i, dx in enumerate(dx_vals):
            mask = ~np.isnan(V_pnp[i, :])
            color = get_dx_color(dx)
            ls, marker = get_dx_style(dx)
            label_str = format_dx_to_latex_exp(dx)
            scatter_plot = ax_p2.scatter(x_micro[mask], V_pnp[i, mask], color=color, label=label_str)
            d_handles.append((dx, scatter_plot, label_str))

    ax_p2.tick_params(axis="both", which="major", length=2*MAG, width=1*MAG)
    ax_p2.tick_params(axis="both", which="minor", length=1*MAG, width=0.3*MAG)
    ax_p2.tick_params(axis="both", labelsize=BASE_FONTS["ytick"] * MAG * 1.4)

    c_handles.sort(key=lambda item: item[0])
    ax_p2.set_ylim(-0.06, 0)
    ax_p2.set_ylabel(r"$\Psi_{\mathrm{PNP}}$ ($\mathrm{V}$)", fontsize=BASE_FONTS["axes_label"] * MAG * 2.2)

    # ========================================================
    # f) Zoom Potential PNP
    # ========================================================
    zoom_handles_f = []
    for j, path in enumerate(on_files):
        if not os.path.exists(path):
            continue
        _, _, V_pnp, _, dx_vals, x_phys, _ = read_pretty_table(path)
        x_micro = x_phys * 1e6
        for i, dx in enumerate(dx_vals):
            mask = ~np.isnan(V_pnp[i, :])
            color = get_dx_color(dx)
            ls, marker = get_dx_style(dx)
            label_str = format_dx_to_latex_exp(dx)
            scatter_plot = ax_z2.scatter(x_micro[mask], V_pnp[i, mask], color=color, lw=3.8, ls=ls, label=label_str)
            zoom_handles_f.append((dx, scatter_plot, label_str))

    ax_z2.tick_params(axis="both", which="major", length=2*MAG, width=1*MAG)
    ax_z2.tick_params(axis="both", which="minor", length=1*MAG, width=0.3*MAG)
    ax_z2.tick_params(axis="both", labelsize=BASE_FONTS["ytick"] * MAG * 1.4)

    zoom_handles_f.sort(key=lambda item: item[0])
    ax_z2.set_xlim(0.1, 0.2)
    ax_z2.set_ylim(-0.03, -0.01)
    ax_z2.set_xticks([0.10, 0.15, 0.20])
    ax_z2.set_yticks([-0.03, -0.02, -0.01])
    ax_z2.set_ylabel(r"$\Psi_{\mathrm{PNP}}$ ($\mathrm{V}$)", fontsize=BASE_FONTS["axes_label"] * MAG * 2.2)

    # Unified x axes labels
    for ax in [ax_c2, ax_p2, ax_z2]:
        ax.set_xlabel(r"$x$ ($\mathrm{m}$)", fontsize=BASE_FONTS["axes_label"] * MAG * 2.2)

    # Subplot labels annotation
    for r_idx in range(2):
        for c_idx in range(3):
            axes[r_idx, c_idx].text(
                0.04, 0.04, subplot_labels[r_idx][c_idx],
                transform=axes[r_idx, c_idx].transAxes,
                fontsize=17.5, fontweight="bold", va="bottom", ha="left"
            )

    plt.tight_layout()

    if save:
        base = f"{filename_base}_BC_{BCs}"
        os.makedirs(OUTDIR, exist_ok=True)
        for fmt in SAVE_FORMATS:
            path_to_save = os.path.join(OUTDIR, f"{base}.{fmt}")
            plt.savefig(path_to_save, dpi=600 if fmt == "pdf" else 400, bbox_inches="tight")

    if SHOW_PLOTS:
        plt.show()

    plt.close(fig)

def main():
    plot_unified_composite(RESULTS_ON_FILES, RESULTS_OFF_FILES)

if __name__ == "__main__":
    main()
