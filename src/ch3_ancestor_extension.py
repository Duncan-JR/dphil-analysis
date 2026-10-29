"""A complete-data toy example of ancestor extension in tsinfer 0.4/0.5.

The two small replays follow ``AncestorBuilder.compute_ancestral_states`` in
https://github.com/tskit-dev/tsinfer/blob/0.4.1/tsinfer/algorithm.py and
https://github.com/Duncan-JR/tsinfer/blob/0.5.x/tsinfer/algorithm.py
(backport commit d694eea). The backport gates both removal and flag updates.
Site times use derived allele counts, the default age proxy. The notebook
supplies the example matrix; no evaluation data are needed.
"""

import dataclasses
import pathlib

import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np


@dataclasses.dataclass
class FigureConfig:
    genotypes: tuple[tuple[int, ...], ...]
    samples: tuple[str, ...] = tuple("adbcfg")
    focal_site: int = 1  # One-indexed, as in the drawing.
    versions: tuple[str, str] = ("0.4.1", "0.5.1")
    version_colors: tuple[str, str] = ("#1a80bb", "#ea801c")
    sample_color: str = "#748a98"
    focal_color: str = "#cc3838"
    conflict_color: str = "#ff2626"
    conflict_line_width: float = 1.8
    first_conflict_dashes: int = 3
    focal_ring_width: float = 0.065
    allele_text_offset: float = 0.038
    outside_color: str = "#d6dfe5"
    ink_color: str = "#253642"
    muted_color: str = "#66747d"
    contained_color: str = "#8167a6"
    older_allele_color: str = "#ead4a5"
    neutral_allele_color: str = "#d7dce0"
    figsize: tuple[float, float] = (9.0, 4.9)
    legend_width: float = 6.3
    site_spacing: float = 1.4
    row_spacing: float = 0.76
    outside_gap: float = 0.22
    bar_height: float = 0.52
    circle_radius: float = 0.205
    allele_fontsize: float = 10.8
    label_fontsize: float = 11
    legend_fontsize: float = 10
    dpi: int = 400


@dataclasses.dataclass
class ExtensionTrace:
    haplotype: np.ndarray
    marks: dict[int, list[int]]
    removals: dict[int, int]
    start: int
    end: int


def trace_ancestor(config, version):
    """Replay the selected algorithm on complete binary data, recording conflicts.

    Indexes in the returned trace are zero-based. Younger sites are assigned 0
    even when their sample-set consensus is 1. In the 0.5.x backport both
    removal and flag updates require an older site or a derived carrier
    outside the *current* sample set.
    """
    if version not in config.versions:
        raise ValueError(f"Unknown version: {version}")
    genotypes = np.asarray(config.genotypes, dtype=np.int8)
    if not np.all((genotypes == 0) | (genotypes == 1)):
        raise ValueError("The illustration requires complete binary genotypes")
    if genotypes.shape[0] != len(config.samples):
        raise ValueError("Each sample needs one genotype row")
    counts = genotypes.sum(axis=0)
    focal = config.focal_site - 1
    initial = set(np.flatnonzero(genotypes[:, focal]))
    threshold = len(initial) // 2
    haplotype = np.full(genotypes.shape[1], -1, dtype=np.int8)
    haplotype[focal] = 1
    marks = {sample: [] for sample in initial}
    removals = {}
    boundaries = []
    directions = (range(focal - 1, -1, -1), range(focal + 1, len(counts)))
    for sites in directions:
        sample_set = initial.copy()
        disagree = set()
        last = focal
        for site in sites:
            last = site
            alleles = genotypes[:, site]
            ones = sum(alleles[sample] == 1 for sample in sample_set)
            consensus = int(2 * ones >= len(sample_set))
            older = counts[site] > counts[focal]
            haplotype[site] = consensus if older else 0
            informative = older or counts[site] > ones
            if older or (version == "0.5.1" and informative):
                removed = {
                    sample for sample in sample_set & disagree
                    if alleles[sample] != consensus
                }
                for sample in removed:
                    removals[sample] = site
                sample_set -= removed
            if older or (version == "0.5.1" and informative):
                disagree = {sample for sample in sample_set if alleles[sample] != consensus}
                for sample in disagree:
                    marks[sample].append(site)
            if len(sample_set) <= threshold:
                break
        boundaries.append(last)
    return ExtensionTrace(haplotype, marks, removals, boundaries[0], boundaries[1] + 1)


def _circle(ax, x, y, allele, color, edge, config, radius=None):
    """A flat allele symbol keeps the small genotype matrix easy to read."""
    if radius is None:
        radius = config.circle_radius
    ax.add_patch(mpatches.Circle(
        (x, y), radius, facecolor=color, edgecolor=edge,
        linewidth=1.3 if edge == config.focal_color else 0.7, zorder=5,
    ))
    ax.text(x, y + config.allele_text_offset, str(allele), ha="center", va="center",
            fontsize=config.allele_fontsize,
            color=config.ink_color, zorder=6)


def _first_conflict(ax, x, y, color, config):
    """Equal solid segments at both ends, shared by the matrix and legend."""
    boundaries = np.linspace(
        y - config.bar_height / 2, y + config.bar_height / 2,
        2 * config.first_conflict_dashes,
    )
    for start, end in zip(boundaries[::2], boundaries[1::2]):
        line, = ax.plot([x, x], [start, end], color=color,
                        lw=config.conflict_line_width, solid_capstyle="butt", zorder=8)
        clip = mpatches.Rectangle(
            (x - 0.15, y - config.bar_height / 2), 0.3, config.bar_height,
            transform=ax.transData,
        )
        line.set_clip_path(clip)


def _ragged(ax, x, y, color, config):
    ys = np.linspace(y - config.bar_height / 2, y + config.bar_height / 2, 9)
    xs = x + np.tile([-0.03, 0.03], 5)[:len(ys)]
    line, = ax.plot(xs, ys, color=color, lw=config.conflict_line_width,
                    solid_capstyle="butt", zorder=8)
    # Clip the stroke itself, including its angled caps, to the bar height.
    clip = mpatches.Rectangle(
        (x - 0.15, y - config.bar_height / 2), 0.3, config.bar_height,
        transform=ax.transData,
    )
    line.set_clip_path(clip)


def _bar(ax, left, right, y, color, config, edge="none", hatch=None):
    patch = mpatches.Rectangle(
        (left, y - config.bar_height / 2), right - left, config.bar_height,
        facecolor=color, edgecolor=edge, linewidth=0.5, hatch=hatch, zorder=1,
    )
    ax.add_patch(patch)


def _site_bracket(ax, sites, label, color, config):
    left = min(sites) * config.site_spacing - 0.31
    right = max(sites) * config.site_spacing + 0.31
    middle = (left + right) / 2
    y = -1.21
    vertices = [
        (left, y + 0.12),
        (left, y + 0.02), (left + 0.06, y), (left + 0.15, y),
        (middle - 0.15, y), (middle - 0.04, y), (middle, y - 0.12),
        (middle + 0.04, y), (middle + 0.15, y), (right - 0.15, y),
        (right - 0.06, y), (right, y + 0.02), (right, y + 0.12),
    ]
    codes = [mpath.Path.MOVETO] + [mpath.Path.CURVE4] * 12
    ax.add_patch(mpatches.PathPatch(mpath.Path(vertices, codes),
                                  facecolor="none", edgecolor=color, linewidth=1))
    ax.text(middle, -1.48, label, color=color, ha="center",
            va="bottom", fontsize=config.legend_fontsize)


def _legend(ax, x, config):
    fontsize = config.legend_fontsize
    ink = config.ink_color
    ax.text(x, -0.05, r"Sample set ($S$)", fontsize=fontsize, fontweight="bold", color=ink)
    entries = (
        (0.45, config.sample_color, "Retained (both versions)"),
        (1.00, config.version_colors[0], "Retained (0.4 only)"),
        (1.55, config.outside_color, "Outside / removed"),
    )
    for y, color, label in entries:
        _bar(ax, x, x + 0.42, y, color, config)
        ax.text(x + 0.58, y, label, va="center", color=ink, fontsize=fontsize)
    ax.text(x, 2.40, "Consensus conflicts", fontsize=fontsize,
            fontweight="bold", color=ink)
    for y in (2.90, 3.45):
        _bar(ax, x, x + 0.42, y, config.sample_color, config)
    _first_conflict(ax, x + 0.21, 2.90, config.conflict_color, config)
    ax.text(x + 0.58, 2.90, "First disagreement", va="center", fontsize=fontsize, color=ink)
    _ragged(ax, x + 0.21, 3.45, config.conflict_color, config)
    ax.text(x + 0.58, 3.45, "Second: remove", va="center", fontsize=fontsize, color=ink)
    ax.text(x, 4.10, "Red: both versions", va="center", fontsize=fontsize, color=config.conflict_color)
    ax.text(x, 4.60, "Orange: 0.5 only", va="center", fontsize=fontsize, color=config.version_colors[1])
    ax.text(x, 5.35, "Highlighted sites", fontsize=fontsize, fontweight="bold", color=ink)
    sites = (
        (5.90, config.focal_color, "Focal Site"),
        (6.45, config.contained_color, "Within-clade doubleton"),
        (7.00, config.version_colors[1], "Conflicting-clade doubleton"),
    )
    for y, color, label in sites:
        _circle(ax, x + 0.21, y, 1, color, color, config, radius=0.17)
        ax.text(x + 0.58, y, label, va="center", color=ink, fontsize=fontsize)


def plot_ancestor_extension(config, figure_folder=None):
    """Display the toy comparison and optionally export PNG and editable SVG.

    Arrows mean that the stopping criterion has not been reached within the
    displayed window. The stop site itself is included in the inferred 0.5
    ancestor. The eight-site example removes b in both versions, then c only
    in 0.5, reaching the halving threshold and leaving 0.4 two sites longer.
    """
    genotypes = np.asarray(config.genotypes, dtype=np.int8)
    counts = genotypes.sum(axis=0)
    old = trace_ancestor(config, config.versions[0])
    new = trace_ancestor(config, config.versions[1])
    focal = config.focal_site - 1
    initial = set(np.flatnonzero(genotypes[:, focal]))
    contained = []
    shared = []
    for site in np.flatnonzero(counts == 2):
        carriers = set(np.flatnonzero(genotypes[:, site]))
        if carriers <= initial:
            contained.append(site + 1)
        elif carriers & initial:
            shared.append(site + 1)

    fig, ax = plt.subplots(figsize=config.figsize)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
    site_positions = np.arange(1, genotypes.shape[1] + 1) * config.site_spacing
    left_x = 0.5 * config.site_spacing
    last_x = (genotypes.shape[1] + 0.5) * config.site_spacing
    ys = np.arange(len(config.samples), dtype=float) * config.row_spacing
    ys[len(initial):] += config.outside_gap
    axis_y = ys[-1] + 0.88
    ancestor_ys = (axis_y + 1.06, axis_y + 1.93)
    ax.set(xlim=(-2.55, last_x + config.legend_width), ylim=(ancestor_ys[-1] + 0.63, -2.15))
    ax.set_aspect("equal")
    ax.axis("off")

    ax.text(0.12, -0.76, "Allele count", ha="right", va="center",
            color=config.ink_color, fontsize=config.label_fontsize)
    for site, (x, count) in enumerate(zip(site_positions, counts), start=1):
        color = config.ink_color
        if site == config.focal_site:
            color = config.focal_color
        elif site in contained:
            color = config.contained_color
        elif site in shared:
            color = config.version_colors[1]
        ax.text(x, -0.76, str(count), color=color, ha="center", va="center",
                fontsize=config.label_fontsize, fontweight="bold" if count > counts[focal] else "normal")
    _site_bracket(ax, [config.focal_site], "Focal", config.focal_color, config)
    _site_bracket(ax, contained, "Within\nclade", config.contained_color, config)
    _site_bracket(ax, shared, "Conflicting\nclade", config.version_colors[1], config)

    for sample, (name, row, y) in enumerate(zip(config.samples, genotypes, ys)):
        _bar(ax, left_x, last_x, y, config.outside_color, config)
        if sample in initial:
            stop = new.removals.get(sample)
            right = last_x if stop is None else (stop + 1.5) * config.site_spacing
            _bar(ax, left_x, right, y, config.sample_color, config)
            if sample in new.removals and sample not in old.removals:
                _bar(ax, right, last_x, y, config.version_colors[0], config)
        ax.text(0.18, y, name, ha="right", va="center", fontsize=config.label_fontsize + 1,
                fontstyle="italic", color=config.focal_color if sample in initial else config.ink_color,
                fontweight="bold" if sample in initial else "normal")
        for site, allele in enumerate(row):
            face = config.neutral_allele_color if allele else "#edf0f2"
            edge = "#aab4bb"
            if allele:
                if site == focal:
                    face, edge = config.focal_color, config.focal_color
                elif site + 1 in contained:
                    face, edge = config.contained_color, config.contained_color
                elif site + 1 in shared:
                    face, edge = config.version_colors[1], config.version_colors[1]
                elif counts[site] > counts[focal]:
                    face, edge = config.older_allele_color, "#b59a66"
            _circle(ax, site_positions[site], y, allele, face, edge, config)
        if sample in new.removals:
            specific = sample not in old.removals
            trace = new if specific else old
            removal = trace.removals[sample]
            color = config.version_colors[1] if specific else config.conflict_color
            first = max(site for site in trace.marks[sample] if site < removal)
            _first_conflict(ax, (first + 1.5) * config.site_spacing, y, color, config)
            _ragged(ax, (removal + 1.5) * config.site_spacing, y, color, config)

    ax.text(-0.90, (ys[0] + ys[-1]) / 2, "Sample haplotype", rotation=90,
            ha="center", va="center", fontsize=config.label_fontsize, color=config.ink_color)
    ax.plot([left_x, last_x], [axis_y, axis_y], lw=0.8, color="#8e9aa2")
    ax.text(0.12, axis_y + 0.19, "Site", ha="right", va="center",
            fontsize=config.label_fontsize, color=config.ink_color)
    for site, x in enumerate(site_positions, start=1):
        ax.plot([x, x], [axis_y - 0.06, axis_y + 0.08], color="#8e9aa2", lw=0.8)
        ax.text(x, axis_y + 0.30, str(site), ha="center", va="center",
                fontsize=config.legend_fontsize, color=config.ink_color)

    for trace, y, version, color in zip((old, new), ancestor_ys, config.versions, config.version_colors):
        stop_x = (trace.end + 0.4) * config.site_spacing
        _bar(ax, left_x, stop_x, y, color, config)
        if trace.end == genotypes.shape[1]:
            ax.add_patch(mpatches.Polygon(
                [[stop_x, y - config.bar_height / 2], [stop_x + 0.32, y],
                 [stop_x, y + config.bar_height / 2]], facecolor=color, edgecolor="none", zorder=1,
            ))
        else:
            ax.plot([stop_x, stop_x], [y - 0.34, y + 0.34], color=color, lw=1.5)
        short_version = ".".join(version.split(".")[:2])
        ax.text(-0.04, y, f"Inferred ancestor\n({short_version})", ha="right", va="center",
                fontsize=config.label_fontsize - 0.5, color=color, linespacing=1.35)
        for site in range(trace.start, trace.end):
            if site == focal:
                # The full-size white button covers the inner part of the red
                # disk, so the thicker ring grows entirely outwards.
                ax.add_patch(mpatches.Circle(
                    (site_positions[site], y), config.circle_radius + config.focal_ring_width,
                    facecolor=config.focal_color, edgecolor="none", zorder=3,
                ))
            _circle(ax, site_positions[site], y, trace.haplotype[site], "#ffffff", "#ffffff", config)

    legend_x = last_x + 0.62
    _legend(ax, legend_x, config)
    if figure_folder is not None:
        folder = pathlib.Path(figure_folder).expanduser()
        folder.mkdir(parents=True, exist_ok=True)
        for suffix in ("png", "svg"):
            fig.savefig(folder / f"ancestor_extension_comparison.{suffix}", dpi=config.dpi,
                        bbox_inches="tight", facecolor="white")
    plt.show()
    plt.close(fig)
    return fig
