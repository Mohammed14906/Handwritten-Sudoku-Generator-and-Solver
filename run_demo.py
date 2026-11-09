"""Integrated demo: generate puzzle, introduce misreads, read with NN, correct and solve.

Produces a step-by-step visual walkthrough suitable for an Intelligent Systems
project showcase.  Each pipeline stage gets its own annotated figure saved to
experiments/step_*.png, plus a combined summary panel.
"""

import argparse
import time
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

from src.data import load_splits
from src import nn as nn_module
from src.puzzle_gen import generate_puzzle, introduce_misreads, generate_mosaic, mosaic_to_grid
from src.csp import find_conflicts
from src.integration import solve_from_mosaic

# ── colour palette ──────────────────────────────────────────────────────────
BG       = "#f7f7f8"
GRID_CLR = "#23272e"
ACCENT   = "#4a90d9"
CORR_CLR = "#e74c3c"
OK_CLR   = "#27ae60"
EMPTY_CLR = "#d5d8dc"
CONF_CMAP = LinearSegmentedColormap.from_list(
    "conf", ["#e74c3c", "#f5b041", "#27ae60"], N=256
)

OUT_DIR = "experiments"


# ═══════════════════════════════════════════════════════════════════════════
#  Grid drawing helpers
# ═══════════════════════════════════════════════════════════════════════════

def _draw_grid_lines(ax):
    """Draw thin and bold Sudoku grid lines on a 9×9 axis."""
    for i in range(10):
        lw = 2.5 if i % 3 == 0 else 0.5
        ax.axhline(i - 0.5, color=GRID_CLR, linewidth=lw)
        ax.axvline(i - 0.5, color=GRID_CLR, linewidth=lw)
    ax.set_xlim(-0.5, 8.5)
    ax.set_ylim(8.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])


def _draw_digit_grid(ax, grid, title="", colors=None, highlights=None,
                     sub_text=None, empty_marker=False):
    """Render a 9×9 digit grid on *ax*.

    - colors: (9,9) array → cell background via CONF_CMAP
    - highlights: dict {(r,c): edgecolor} for rectangles
    - sub_text: dict {(r,c): str} small text below digit
    """
    ax.set_facecolor(BG)

    if colors is not None:
        for r in range(9):
            for c in range(9):
                ax.add_patch(plt.Rectangle(
                    (c - 0.5, r - 0.5), 1, 1,
                    facecolor=CONF_CMAP(colors[r, c]), alpha=0.35,
                    edgecolor="none"
                ))

    for r in range(9):
        for c in range(9):
            val = grid[r, c]
            if val == 0:
                if empty_marker:
                    ax.text(c, r, "·", ha="center", va="center",
                            fontsize=14, color=EMPTY_CLR, fontweight="bold")
                continue
            fc = GRID_CLR
            if highlights and (r, c) in highlights:
                fc = highlights[(r, c)]
            ax.text(c, r, str(val), ha="center", va="center",
                    fontsize=13, color=fc, fontweight="bold",
                    fontfamily="monospace")
            if sub_text and (r, c) in sub_text:
                ax.text(c, r + 0.35, sub_text[(r, c)], ha="center",
                        va="center", fontsize=6, color="#7f8c8d",
                        fontfamily="monospace")

    if highlights:
        for (r, c), ec in highlights.items():
            ax.add_patch(plt.Rectangle(
                (c - 0.5, r - 0.5), 1, 1,
                fill=False, edgecolor=ec, linewidth=2.2
            ))

    _draw_grid_lines(ax)
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", pad=8)


# ═══════════════════════════════════════════════════════════════════════════
#  Per-step visualisation functions
# ═══════════════════════════════════════════════════════════════════════════

def _vis_step0_puzzle(puzzle, solution, misread_log, corrupted, args):
    """Step 0 — puzzle generation & misread injection."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor(BG)

    _draw_digit_grid(axes[0], solution, title="Full Solution (hidden)",
                     empty_marker=False)

    _draw_digit_grid(axes[1], puzzle,
                     title=f"Puzzle ({args.n_givens} givens, seed={args.seed})",
                     empty_marker=True)

    misread_hl = {}
    misread_sub = {}
    for r, c, true_d, wrong_d in misread_log:
        misread_hl[(r, c)] = CORR_CLR
        misread_sub[(r, c)] = f"was {true_d}"
    _draw_digit_grid(axes[2], corrupted,
                     title=f"After {len(misread_log)} Misread(s) Injected",
                     highlights=misread_hl, sub_text=misread_sub,
                     empty_marker=True)

    legend_els = [
        mpatches.Patch(edgecolor=CORR_CLR, facecolor="none", linewidth=2,
                       label="Misread cell"),
    ]
    axes[2].legend(handles=legend_els, loc="upper left", fontsize=8,
                   framealpha=0.9)

    plt.suptitle("Step 0 · Puzzle Generation & Misread Injection",
                 fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(OUT_DIR, "step0_puzzle.png")
    plt.savefig(path, dpi=150, facecolor=BG)
    plt.close()
    print(f"  Saved {path}")


def _vis_step1_mosaic(mosaic, corrupted):
    """Step 1 — mosaic image that the NN will read."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.patch.set_facecolor(BG)

    axes[0].imshow(mosaic, cmap="gray")
    for i in range(10):
        lw = 2 if i % 3 == 0 else 0.5
        axes[0].axhline(i * 28 - 0.5, color="white", linewidth=lw, alpha=0.6)
        axes[0].axvline(i * 28 - 0.5, color="white", linewidth=lw, alpha=0.6)
    axes[0].set_title("Input Mosaic (252 × 252)", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    sample_cells = []
    for r in range(9):
        for c in range(9):
            if corrupted[r, c] != 0:
                sample_cells.append((r, c))
    sample_cells = sample_cells[:6]

    for idx, (r, c) in enumerate(sample_cells):
        inset = fig.add_axes([0.55 + (idx % 3) * 0.14,
                              0.55 - (idx // 3) * 0.42,
                              0.12, 0.35])
        cell_img = mosaic[r*28:(r+1)*28, c*28:(c+1)*28]
        inset.imshow(cell_img, cmap="gray")
        inset.set_title(f"({r},{c})={corrupted[r,c]}", fontsize=9)
        inset.axis("off")

    axes[1].axis("off")
    axes[1].set_title("Sample Cells Fed to NN", fontsize=12, fontweight="bold")

    plt.suptitle("Step 1 · Mosaic Generation (MNIST Digit Images)",
                 fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(OUT_DIR, "step1_mosaic.png")
    plt.savefig(path, dpi=150, facecolor=BG)
    plt.close()
    print(f"  Saved {path}")


def _vis_step2_nn(data):
    """Step 2 — NN digit recognition with confidence heatmap."""
    grid = data["grid"]
    confidences = data["confidences"]
    top2 = data["top2_digits"]
    top2_p = data["top2_probs"]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.patch.set_facecolor(BG)

    # Panel 1: predicted grid with confidence colouring
    sub = {}
    for r in range(9):
        for c in range(9):
            if grid[r, c] != 0:
                sub[(r, c)] = f"{confidences[r,c]:.0%}"
    _draw_digit_grid(axes[0], grid,
                     title="NN Predictions (colour = confidence)",
                     colors=confidences, sub_text=sub, empty_marker=True)

    sm = plt.cm.ScalarMappable(cmap=CONF_CMAP, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[0], fraction=0.046, pad=0.04)
    cbar.set_label("Confidence", fontsize=9)

    # Panel 2: top-2 alternative per cell (where 2nd choice > 5 %)
    alt_grid = grid.copy()
    alt_hl = {}
    alt_sub = {}
    for r in range(9):
        for c in range(9):
            idx = r * 9 + c
            if grid[r, c] == 0:
                continue
            p2 = top2_p[idx, 1]
            if p2 > 0.05:
                alt_hl[(r, c)] = "#e67e22"
                alt_sub[(r, c)] = f"alt {top2[idx,1]}@{p2:.0%}"
    _draw_digit_grid(axes[1], alt_grid,
                     title="Cells With Notable 2nd Choice (>5 %)",
                     highlights=alt_hl, sub_text=alt_sub, empty_marker=True)

    # Panel 3: confidence histogram
    confs = confidences[grid != 0].flatten()
    axes[2].hist(confs, bins=20, color=ACCENT, edgecolor="white", alpha=0.85)
    axes[2].axvline(confs.mean(), color=CORR_CLR, linestyle="--", linewidth=1.5,
                    label=f"mean = {confs.mean():.2%}")
    axes[2].axvline(np.median(confs), color=OK_CLR, linestyle="--", linewidth=1.5,
                    label=f"median = {np.median(confs):.2%}")
    axes[2].set_xlabel("Confidence", fontsize=10)
    axes[2].set_ylabel("Cell count", fontsize=10)
    axes[2].set_title("Confidence Distribution", fontsize=12, fontweight="bold")
    axes[2].legend(fontsize=9)
    axes[2].set_facecolor(BG)

    n_digit = len(data["digit_cells"])
    n_blank = len(data["blank_cells"])
    plt.suptitle(f"Step 2 · Neural Network Reading  "
                 f"({n_digit} digits recognised, {n_blank} blanks detected)",
                 fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(OUT_DIR, "step2_nn_reading.png")
    plt.savefig(path, dpi=150, facecolor=BG)
    plt.close()
    print(f"  Saved {path}")


def _vis_step3_conflicts(data):
    """Step 3 — conflict detection via AC-3."""
    grid = data["grid"]
    conflicts = data["conflicts"]
    units = data["conflict_units"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor(BG)

    # Panel 1: grid with conflict cells highlighted
    hl = {(r, c): CORR_CLR for (r, c) in conflicts}
    sub = {}
    for (r, c), u_list in units.items():
        sub[(r, c)] = ", ".join(u_list)
    _draw_digit_grid(axes[0], grid,
                     title=f"Conflicts Detected: {len(conflicts)} cell(s)",
                     highlights=hl, sub_text=sub, empty_marker=True)

    legend_els = [
        mpatches.Patch(edgecolor=CORR_CLR, facecolor="none", linewidth=2,
                       label="Conflicting cell"),
    ]
    axes[0].legend(handles=legend_els, loc="lower left", fontsize=8,
                   framealpha=0.9)

    # Panel 2: text summary table
    axes[1].axis("off")
    axes[1].set_facecolor(BG)
    lines = [
        f"Total conflicting cells:  {len(conflicts)}",
        "",
        "Details per cell:",
        "─" * 40,
    ]
    for (r, c) in sorted(conflicts):
        digit = grid[r, c]
        u_str = ", ".join(units.get((r, c), []))
        lines.append(f"  ({r},{c}) = {digit}   violates: {u_str}")

    if not conflicts:
        lines.append("  No conflicts — NN reading is consistent!")

    txt = "\n".join(lines)
    axes[1].text(0.05, 0.95, txt, transform=axes[1].transAxes,
                 fontsize=10, fontfamily="monospace", verticalalignment="top",
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                           edgecolor="#bdc3c7"))

    plt.suptitle("Step 3 · Conflict Detection (constraint checking)",
                 fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(OUT_DIR, "step3_conflicts.png")
    plt.savefig(path, dpi=150, facecolor=BG)
    plt.close()
    print(f"  Saved {path}")


def _vis_step4_search(data, nn_grid, probabilities):
    """Step 4 — search-based correction."""
    stats = data["search_stats"]
    corrections = data["corrections_found"]
    strategy = data["strategy"]
    skipped = data.get("skipped", False)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.patch.set_facecolor(BG)

    if skipped or not corrections:
        # No corrections needed
        _draw_digit_grid(axes[0], nn_grid,
                         title="NN Grid (no corrections needed)",
                         empty_marker=True)
        axes[1].axis("off")
        axes[1].text(0.5, 0.5, "No conflicts found —\nsearch step skipped.",
                     ha="center", va="center", fontsize=14, color=OK_CLR,
                     fontweight="bold", transform=axes[1].transAxes)
        axes[2].axis("off")
    else:
        # Panel 1: before (NN grid with conflict highlights)
        conflicts_before = find_conflicts(nn_grid)
        hl_before = {(r, c): CORR_CLR for (r, c) in conflicts_before}
        _draw_digit_grid(axes[0], nn_grid, title="Before (NN reading)",
                         highlights=hl_before, empty_marker=True)

        # Panel 2: after corrections
        corrected = nn_grid.copy()
        hl_after = {}
        sub_after = {}
        for (r, c, new_d) in corrections:
            old_d = nn_grid[r, c]
            corrected[r, c] = new_d
            hl_after[(r, c)] = OK_CLR
            idx = r * 9 + c
            old_p = probabilities[idx, old_d - 1]
            new_p = probabilities[idx, new_d - 1]
            sub_after[(r, c)] = f"{old_d}→{new_d}"
        _draw_digit_grid(axes[1], corrected, title="After Corrections",
                         highlights=hl_after, sub_text=sub_after,
                         empty_marker=True)

        legend_els = [
            mpatches.Patch(edgecolor=CORR_CLR, facecolor="none", linewidth=2,
                           label="Conflict (before)"),
            mpatches.Patch(edgecolor=OK_CLR, facecolor="none", linewidth=2,
                           label="Corrected cell"),
        ]
        axes[1].legend(handles=legend_els, loc="lower left", fontsize=8,
                       framealpha=0.9)

        # Panel 3: search statistics
        axes[2].axis("off")
        axes[2].set_facecolor(BG)
        lines = [
            f"Search Strategy:     {strategy}",
            f"Nodes Expanded:      {stats['nodes_expanded']}",
            f"Nodes Generated:     {stats['nodes_generated']}",
            f"Solution Cost:       {stats['solution_cost']:.6f}",
            f"Max Corrections:     {data['max_corrections']}",
            f"Corrections Made:    {len(corrections)}",
            "",
            "Correction Details:",
            "─" * 44,
        ]
        for (r, c, new_d) in corrections:
            old_d = nn_grid[r, c]
            idx = r * 9 + c
            old_p = probabilities[idx, old_d - 1]
            new_p = probabilities[idx, new_d - 1]
            cost = max(old_p - new_p, 1e-6)
            lines.append(
                f"  ({r},{c}): {old_d}→{new_d}  "
                f"P({old_d})={old_p:.3f}  P({new_d})={new_p:.3f}  "
                f"cost={cost:.4f}"
            )

        txt = "\n".join(lines)
        axes[2].text(0.05, 0.95, txt, transform=axes[2].transAxes,
                     fontsize=10, fontfamily="monospace",
                     verticalalignment="top",
                     bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                               edgecolor="#bdc3c7"))

    plt.suptitle(f"Step 4 · Search-Based Correction ({strategy})",
                 fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(OUT_DIR, "step4_search.png")
    plt.savefig(path, dpi=150, facecolor=BG)
    plt.close()
    print(f"  Saved {path}")


def _vis_step5_csp(data, nn_grid, corrections):
    """Step 5 — CSP solving."""
    csp_stats = data["csp_stats"]
    method = data["method"]
    solved = data["solved"]
    grid_in = data["grid_in"]
    solution = data.get("solution")
    n_empty = data.get("empty_cells", 0)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.patch.set_facecolor(BG)

    # Panel 1: grid fed to CSP (givens + corrected, blanks marked)
    hl_given = {}
    for r in range(9):
        for c in range(9):
            if grid_in[r, c] != 0:
                hl_given[(r, c)] = ACCENT
    _draw_digit_grid(axes[0], grid_in,
                     title=f"CSP Input ({81 - n_empty} given, {n_empty} empty)",
                     highlights=hl_given, empty_marker=True)

    # Panel 2: solved grid
    if solved and solution is not None:
        filled_hl = {}
        for r in range(9):
            for c in range(9):
                if grid_in[r, c] == 0 and solution[r, c] != 0:
                    filled_hl[(r, c)] = OK_CLR
        _draw_digit_grid(axes[1], solution,
                         title="CSP Solution",
                         highlights=filled_hl, empty_marker=False)
        legend_els = [
            mpatches.Patch(edgecolor=ACCENT, facecolor="none", linewidth=2,
                           label="Given / NN digit"),
            mpatches.Patch(edgecolor=OK_CLR, facecolor="none", linewidth=2,
                           label="Filled by CSP"),
        ]
        axes[1].legend(handles=legend_els, loc="lower left", fontsize=8,
                       framealpha=0.9)
    else:
        axes[1].axis("off")
        axes[1].text(0.5, 0.5, "No solution found.",
                     ha="center", va="center", fontsize=16, color=CORR_CLR,
                     fontweight="bold", transform=axes[1].transAxes)

    # Panel 3: CSP statistics
    axes[2].axis("off")
    axes[2].set_facecolor(BG)
    lines = [
        f"Method:            {method}",
        f"Solved:            {'Yes' if solved else 'No'}",
        f"Assignments:       {csp_stats.get('assignments', '—')}",
        f"Backtracks:        {csp_stats.get('backtracks', '—')}",
        f"Arc Checks:        {csp_stats.get('arc_checks', '—')}",
        f"Empty Cells:       {n_empty}",
    ]
    if solved:
        lines += [
            "",
            "Technique Breakdown:",
            "─" * 40,
            "  BT  = Backtracking with MRV heuristic",
            "  FC  = Forward Checking (prune neighbors)",
            "  AC3 = Arc Consistency preprocessing",
        ]
    txt = "\n".join(lines)
    axes[2].text(0.05, 0.95, txt, transform=axes[2].transAxes,
                 fontsize=10, fontfamily="monospace",
                 verticalalignment="top",
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                           edgecolor="#bdc3c7"))

    plt.suptitle(f"Step 5 · CSP Solving ({method})",
                 fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(OUT_DIR, "step5_csp.png")
    plt.savefig(path, dpi=150, facecolor=BG)
    plt.close()
    print(f"  Saved {path}")


def _vis_summary(mosaic, puzzle, solution, nn_grid, result, misread_log,
                 corrections, elapsed, args):
    """Combined summary panel."""
    fig = plt.figure(figsize=(22, 14))
    fig.patch.set_facecolor(BG)
    gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.30)

    # Row 0: mosaic → NN reading → conflicts → corrected
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(mosaic, cmap="gray")
    for i in range(10):
        lw = 1.5 if i % 3 == 0 else 0.3
        ax0.axhline(i * 28 - 0.5, color="white", linewidth=lw, alpha=0.5)
        ax0.axvline(i * 28 - 0.5, color="white", linewidth=lw, alpha=0.5)
    ax0.set_title("① Mosaic Input", fontsize=10, fontweight="bold")
    ax0.axis("off")

    ax1 = fig.add_subplot(gs[0, 1])
    confs = np.zeros((9, 9))
    for r in range(9):
        for c in range(9):
            if nn_grid[r, c] != 0:
                confs[r, c] = 1.0
    _draw_digit_grid(ax1, nn_grid, title="② NN Reading", empty_marker=True)

    ax2 = fig.add_subplot(gs[0, 2])
    conflict_cells = find_conflicts(nn_grid)
    hl_conf = {(r, c): CORR_CLR for (r, c) in conflict_cells}
    _draw_digit_grid(ax2, nn_grid, title=f"③ Conflicts ({len(conflict_cells)})",
                     highlights=hl_conf, empty_marker=True)

    ax3 = fig.add_subplot(gs[0, 3])
    corrected_grid = nn_grid.copy()
    hl_corr = {}
    for entry in result["corrections"]:
        r, c = entry["cell"]
        corrected_grid[r, c] = entry["new_digit"]
        hl_corr[(r, c)] = OK_CLR
    _draw_digit_grid(ax3, corrected_grid, title="④ Corrected",
                     highlights=hl_corr, empty_marker=True)

    # Row 1: CSP input → solution → true solution → match check
    ax4 = fig.add_subplot(gs[1, 0])
    _draw_digit_grid(ax4, corrected_grid, title="⑤ CSP Input",
                     empty_marker=True)

    ax5 = fig.add_subplot(gs[1, 1])
    if result["solved"]:
        filled_hl = {}
        for r in range(9):
            for c in range(9):
                if corrected_grid[r, c] == 0 and result["solution"][r, c] != 0:
                    filled_hl[(r, c)] = OK_CLR
        _draw_digit_grid(ax5, result["solution"], title="⑥ CSP Solution",
                         highlights=filled_hl)
    else:
        ax5.axis("off")
        ax5.text(0.5, 0.5, "UNSOLVED", ha="center", va="center",
                 fontsize=14, color=CORR_CLR, transform=ax5.transAxes)
        ax5.set_title("⑥ CSP Solution", fontsize=10, fontweight="bold")

    ax6 = fig.add_subplot(gs[1, 2])
    _draw_digit_grid(ax6, solution, title="⑦ True Solution")

    ax7 = fig.add_subplot(gs[1, 3])
    ax7.axis("off")
    ax7.set_facecolor(BG)
    match = result["solved"] and np.array_equal(result["solution"], solution)
    status = "MATCH ✓" if match else "MISMATCH ✗"
    color = OK_CLR if match else CORR_CLR
    ax7.text(0.5, 0.5, status, ha="center", va="center",
             fontsize=28, fontweight="bold", color=color,
             transform=ax7.transAxes)
    ax7.set_title("⑧ Verification", fontsize=10, fontweight="bold")

    # Row 2: statistics panel
    ax_stats = fig.add_subplot(gs[2, :])
    ax_stats.axis("off")
    ax_stats.set_facecolor(BG)

    ss = result["search_stats"] or {}
    cs = result["csp_stats"] or {}
    stats_text = (
        f"Pipeline Configuration\n"
        f"{'─' * 90}\n"
        f"  Seed: {args.seed}    Givens: {args.n_givens}    "
        f"Misreads injected: {args.misreads}    "
        f"CSP: {args.csp}    Search: {args.search}\n\n"
        f"Neural Network\n"
        f"{'─' * 90}\n"
        f"  Architecture: 784 → 256 → 128 → 9 (ReLU, softmax)    "
        f"Cells classified: {int(np.sum(nn_grid != 0))}    "
        f"Blanks detected: {int(np.sum(nn_grid == 0))}\n\n"
        f"Search Correction ({args.search})\n"
        f"{'─' * 90}\n"
        f"  Nodes expanded: {ss.get('nodes_expanded', 0)}    "
        f"Nodes generated: {ss.get('nodes_generated', 0)}    "
        f"Solution cost: {ss.get('solution_cost', 0):.6f}    "
        f"Corrections: {len(result['corrections'])}\n\n"
        f"CSP Solver ({args.csp})\n"
        f"{'─' * 90}\n"
        f"  Assignments: {cs.get('assignments', '—')}    "
        f"Backtracks: {cs.get('backtracks', '—')}    "
        f"Arc checks: {cs.get('arc_checks', '—')}\n\n"
        f"  Total time: {elapsed:.2f}s    "
        f"Result: {'SOLVED' if result['solved'] else 'UNSOLVED'}    "
        f"Matches ground truth: {'Yes' if match else 'No'}"
    )
    ax_stats.text(0.02, 0.95, stats_text, transform=ax_stats.transAxes,
                  fontsize=9.5, fontfamily="monospace", verticalalignment="top",
                  bbox=dict(boxstyle="round,pad=0.6", facecolor="white",
                            edgecolor="#bdc3c7"))

    plt.suptitle("Handwritten Sudoku Reader & Solver — Full Pipeline Summary",
                 fontsize=15, fontweight="bold", y=0.99)
    path = os.path.join(OUT_DIR, "summary.png")
    plt.savefig(path, dpi=150, facecolor=BG, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# ═══════════════════════════════════════════════════════════════════════════
#  Console helpers
# ═══════════════════════════════════════════════════════════════════════════

def print_grid(grid, title=""):
    if title:
        print(f"\n{title}")
    print("+" + "-" * 7 + "+" + "-" * 7 + "+" + "-" * 7 + "+")
    for r in range(9):
        row_str = "| "
        for c in range(9):
            val = grid[r, c]
            row_str += str(val) if val != 0 else "."
            row_str += " "
            if c % 3 == 2:
                row_str += "| "
        print(row_str)
        if r % 3 == 2:
            print("+" + "-" * 7 + "+" + "-" * 7 + "+" + "-" * 7 + "+")


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Handwritten Sudoku Reader & Solver — Visual Pipeline Demo"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_givens", type=int, default=30)
    parser.add_argument("--misreads", type=int, default=2)
    parser.add_argument("--csp", type=str, default="BT+FC+AC3",
                        choices=["BT", "BT+FC", "BT+FC+AC3"])
    parser.add_argument("--search", type=str, default="A*",
                        choices=["UCS", "Greedy", "A*"])
    parser.add_argument("--model_path", type=str, default="models/nn_model.pkl")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 60)
    print("   Handwritten Sudoku Reader & Solver — Pipeline Demo")
    print("=" * 60)

    # ── Load / train model ──────────────────────────────────────────────
    if os.path.exists(args.model_path):
        print(f"\n[Model] Loading from {args.model_path}")
        model = nn_module.load_model(args.model_path)
    else:
        print("\n[Model] No saved model — training from scratch...")
        X_train, y_train, X_val, y_val, _, _ = load_splits()
        model = nn_module.train(X_train, y_train, X_val, y_val,
                                epochs=30, lr=0.05, batch_size=128, seed=42)
        os.makedirs("models", exist_ok=True)
        nn_module.save_model(model, args.model_path)
        print(f"  Model saved to {args.model_path}")

    # ── Step 0: generate puzzle + misreads ──────────────────────────────
    print(f"\n{'─' * 60}")
    print(f"STEP 0 · Puzzle Generation")
    print(f"{'─' * 60}")
    puzzle, solution = generate_puzzle(args.n_givens, seed=args.seed)
    print(f"  Generated puzzle: {args.n_givens} givens, seed={args.seed}")
    print_grid(puzzle, "  Puzzle:")

    if args.misreads > 0:
        corrupted, misread_log = introduce_misreads(
            puzzle, solution, args.misreads, seed=args.seed
        )
        print(f"\n  Injected {args.misreads} misread(s):")
        for r, c, true_d, wrong_d in misread_log:
            print(f"    Cell ({r},{c}): true={true_d} → misread={wrong_d}")
    else:
        corrupted = puzzle
        misread_log = []

    _vis_step0_puzzle(puzzle, solution, misread_log, corrupted, args)

    # ── Step 1: generate mosaic ─────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print(f"STEP 1 · Mosaic Generation")
    print(f"{'─' * 60}")
    X_train, y_train, *_ = load_splits()
    digit_images = {}
    for d in range(1, 10):
        digit_images[d] = X_train[y_train == d]
    mosaic = generate_mosaic(corrupted, digit_images, seed=args.seed)
    print(f"  Mosaic shape: {mosaic.shape}")
    print(f"  Pixel range: [{mosaic.min():.3f}, {mosaic.max():.3f}]")

    _vis_step1_mosaic(mosaic, corrupted)

    # ── Steps 2-5: solve with callback ──────────────────────────────────
    step_data = {}
    probabilities_holder = [None]

    def on_step(step_name, data):
        step_data[step_name] = data
        if step_name == "nn_reading":
            probabilities_holder[0] = data["probabilities"]

            print(f"\n{'─' * 60}")
            print(f"STEP 2 · Neural Network Reading")
            print(f"{'─' * 60}")
            grid = data["grid"]
            confs = data["confidences"]
            n_digit = len(data["digit_cells"])
            n_blank = len(data["blank_cells"])
            print(f"  Digits classified: {n_digit}")
            print(f"  Blank cells:       {n_blank}")
            given_confs = confs[grid != 0]
            print(f"  Confidence — mean: {given_confs.mean():.2%}, "
                  f"min: {given_confs.min():.2%}, max: {given_confs.max():.2%}")
            low = (given_confs < 0.8).sum()
            if low > 0:
                print(f"  ⚠ {low} cell(s) below 80% confidence")
            print_grid(grid, "  NN Reading:")

            _vis_step2_nn(data)

        elif step_name == "conflict_detection":
            print(f"\n{'─' * 60}")
            print(f"STEP 3 · Conflict Detection")
            print(f"{'─' * 60}")
            conflicts = data["conflicts"]
            units = data["conflict_units"]
            print(f"  Conflicting cells: {len(conflicts)}")
            for (r, c) in sorted(conflicts):
                d = data["grid"][r, c]
                u = ", ".join(units.get((r, c), []))
                print(f"    ({r},{c}) = {d}  → violates {u}")
            if not conflicts:
                print("  No conflicts — NN reading is consistent!")

            _vis_step3_conflicts(data)

        elif step_name == "search_correction":
            print(f"\n{'─' * 60}")
            print(f"STEP 4 · Search Correction ({data['strategy']})")
            print(f"{'─' * 60}")
            stats = data["search_stats"]
            corr = data["corrections_found"]
            skipped = data.get("skipped", False)
            if skipped:
                print("  Skipped (no conflicts).")
            else:
                print(f"  Nodes expanded:  {stats['nodes_expanded']}")
                print(f"  Nodes generated: {stats['nodes_generated']}")
                print(f"  Solution cost:   {stats['solution_cost']:.6f}")
                if corr:
                    print(f"  Corrections ({len(corr)}):")
                    nn_grid = step_data["nn_reading"]["grid"]
                    probs = probabilities_holder[0]
                    for (r, c, new_d) in corr:
                        old_d = nn_grid[r, c]
                        idx = r * 9 + c
                        old_p = probs[idx, old_d - 1]
                        new_p = probs[idx, new_d - 1]
                        cost = max(old_p - new_p, 1e-6)
                        print(f"    ({r},{c}): {old_d} → {new_d}  "
                              f"P({old_d})={old_p:.3f}  P({new_d})={new_p:.3f}  "
                              f"cost={cost:.4f}")
                else:
                    print("  No corrections found / needed.")

            _vis_step4_search(data, step_data["nn_reading"]["grid"],
                              probabilities_holder[0])

        elif step_name == "csp_solve":
            print(f"\n{'─' * 60}")
            print(f"STEP 5 · CSP Solving ({data['method']})")
            print(f"{'─' * 60}")
            cs = data["csp_stats"]
            print(f"  Solved:       {'Yes' if data['solved'] else 'No'}")
            print(f"  Assignments:  {cs.get('assignments', '—')}")
            print(f"  Backtracks:   {cs.get('backtracks', '—')}")
            print(f"  Arc checks:   {cs.get('arc_checks', '—')}")
            n_empty = data.get("empty_cells", 0)
            print(f"  Empty cells filled: {n_empty}")
            if data["solved"] and data.get("solution") is not None:
                print_grid(data["solution"], "  Solution:")

            corrections_found = step_data.get("search_correction", {}).get(
                "corrections_found", frozenset())
            _vis_step5_csp(data, step_data["nn_reading"]["grid"],
                           corrections_found)

    print(f"\n{'─' * 60}")
    print(f"Running full pipeline: CSP={args.csp}, Search={args.search}")
    print(f"{'─' * 60}")

    t0 = time.time()
    result = solve_from_mosaic(mosaic, model, args.csp, args.search,
                               max_corrections=3, callback=on_step)
    elapsed = time.time() - t0

    # ── Final summary ───────────────────────────────────────────────────
    print(f"\n{'═' * 60}")
    print(f"RESULT")
    print(f"{'═' * 60}")
    if result["solved"]:
        matches = np.array_equal(result["solution"], solution)
        print(f"  Solved: Yes")
        print(f"  Matches true solution: {matches}")
    else:
        print(f"  Solved: No — UNREADABLE")
    print(f"  Total time: {elapsed:.2f}s")
    print(f"\nGenerating summary visualisation...")

    nn_grid = step_data["nn_reading"]["grid"]
    corrections = step_data.get("search_correction", {}).get(
        "corrections_found", frozenset())
    _vis_summary(mosaic, puzzle, solution, nn_grid, result,
                 misread_log, corrections, elapsed, args)

    print(f"\nAll figures saved to {OUT_DIR}/")
    print(f"  step0_puzzle.png       — puzzle generation & misread injection")
    print(f"  step1_mosaic.png       — MNIST mosaic image")
    print(f"  step2_nn_reading.png   — neural network predictions + confidence")
    print(f"  step3_conflicts.png    — constraint conflict detection")
    print(f"  step4_search.png       — search-based digit correction")
    print(f"  step5_csp.png          — CSP backtracking solution")
    print(f"  summary.png            — full pipeline overview")


if __name__ == "__main__":
    main()
