"""Chapter 3 figures from ancestor-evaluation CSVs and variant Zarr stores.

The mirrored overshoot histograms adapt the layout in tsinfer-paper's
``lib/visualisation.py``. All data access and plotting here are independent of
that repository. See ``notebooks/ch3_figure_gen.py`` for the figure sequence.
"""

import dataclasses
import logging
import multiprocessing
import pathlib
import re

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import zarr

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class PlotConfig:
    """Settings shared by the chapter figures; lengths in input files are bp."""

    versions: tuple[str, str] = ("0.4.1", "0.5.1")
    version_colors: tuple[str, str] = ("#1a80bb", "#ea801c")
    genotype_scaling_cmap: str = "PRGn"
    genotype_transition_cmap: str = "RdYlBu"
    genotype_color_positions: tuple[float, float] = (0.1, 0.9)
    true_color: str = "#b8b8b8"
    genotype_multipliers: tuple[float, ...] = (1, 2, 5, 10, 100)
    transition_multipliers: tuple[float, ...] = (1, 10)
    mispol_rates: tuple[float, ...] = (0, 0.01, 0.05, 0.1)
    phase_rates: tuple[float, ...] = (0, 0.001, 0.005, 0.01)
    transition_bins: int = 40
    curve_points: int = 256
    histogram_bins: int = 100
    daf_bins: int = 200
    daf_smoothing_bins: float = 1
    dpi: int = 300
    bp_per_mb: float = 1e6
    box_size: tuple[float, float] = (7, 4.5)
    histogram_size: tuple[float, float] = (7, 4.5)
    overshoot_box_size: tuple[float, float] = (8.5, 3.5)
    overshoot_tail_size: tuple[float, float] = (8.5, 4.5)
    overshoot_cmaps: tuple[str, str] = ("Blues", "Oranges")
    version_line_styles: tuple[str, str] = ("-", "--")
    zero_color: str = "#555555"
    quantile_line_style: str = "--"
    quantile_line_width: float = 1.2
    preprocess_chunk_rows: int = 250000
    ancestor_keys: tuple[str, ...] = (
        "replicate", "version", "geno_multiplier", "phase_ser", "mispol_rate", "inferred_node"
    )
    overshoot_box_width_ratio: float = 0.6
    jitter_seed: int = 42
    jitter_max_points: int = 1500
    jitter_width: float = 0.32
    jitter_alpha: float = 0.15
    jitter_size: float = 5
    curve_size: tuple[float, float] = (6, 3.5)
    transition_size: tuple[float, float] = (7, 3.5)
    phase_size: tuple[float, float] = (8, 2.8)
    phase_row_spacing: float = 0.82
    phase_individuals: int = 5
    # GRCh38 chr1 p11.1 ends at 123.4 Mb (UCSC cytoBandIdeo).
    phase_right_mb: float = 123.4
    phase_colors: tuple[str, str] = ("#4477aa", "#ee9944")
    phase_ser: float = 0.005
    workers: int | None = None


@dataclasses.dataclass(frozen=True)
class VariantStore:
    """One replicate and error profile, parsed from its Zarr filename."""

    path: pathlib.Path
    replicate: int
    geno_multiplier: float
    phase_ser: float
    mispol_rate: float


@dataclasses.dataclass
class ChapterData:
    """Ancestor observations and corresponding variant stores for one dataset."""

    overshoots: pd.DataFrame
    spans: pd.DataFrame
    stores: list[VariantStore]
    config: PlotConfig

    def store(self, replicate, geno_multiplier=0, phase_ser=0, mispol_rate=0):
        """Find exactly one profile, including numerically equivalent suffixes."""
        matches = [
            store for store in self.stores
            if store.replicate == replicate
            and store.geno_multiplier == geno_multiplier
            and store.phase_ser == phase_ser
            and store.mispol_rate == mispol_rate
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected one store for rep={replicate}, geno={geno_multiplier}, "
                f"phase={phase_ser}, mispol={mispol_rate}; found {len(matches)}"
            )
        return matches[0]


@dataclasses.dataclass
class GenotypeCounts:
    """Sufficient statistics from one streamed replicate/profile comparison."""

    multiplier: float
    changed: int
    total: int
    homozygous: np.ndarray
    to_heterozygous: np.ndarray


@dataclasses.dataclass
class GenotypeSummary:
    """Pooled rates and conditional transition rates returned by summarisation."""

    rates: pd.DataFrame
    transitions: pd.DataFrame


@dataclasses.dataclass
class GenotypeTask:
    truth: VariantStore
    observed: VariantStore
    bins: int


def _variant_stores(folder, prefix):
    pattern = re.compile(
        r"(.+)-rep(\d+)-geno-([0-9.eE+-]+)-phase([0-9.eE+-]+)"
        r"-mispol([0-9.eE+-]+)\.zarr"
    )
    stores = []
    for path in sorted(folder.glob(f"{prefix}-rep*.zarr")):
        match = pattern.fullmatch(path.name)
        if match is None:
            continue  # The raw stores have no error-profile suffix.
        stores.append(VariantStore(
            path, int(match[2]), float(match[3]), float(match[4]), float(match[5])
        ))
    if not stores:
        raise FileNotFoundError(f"No variant stores for {prefix} in {folder}")
    return stores


@dataclasses.dataclass
class AncestorTask:
    frame: pd.DataFrame
    config: PlotConfig


@dataclasses.dataclass
class AncestorFrames:
    overshoots: pd.DataFrame
    spans: pd.DataFrame


@dataclasses.dataclass(frozen=True)
class AncestorPaths:
    overshoots: pathlib.Path
    spans: pathlib.Path


def _reduce_overshoots(frame, keys):
    # Prefer mispolarised focal associations on both sides. Within that set,
    # select the most negative overshoot independently for each side.
    ranked = frame.sort_values(["focal_site_mispolarised", "overshoot"],
                               ascending=[False, True], kind="stable")
    selected = ranked.drop_duplicates(keys + ["side"])
    columns = keys + ["side", "allele_frequency", "true_focal_site",
                      "focal_site_mispolarised", "true_derived_count",
                      "true_allele_frequency", "overshoot"]
    return selected.loc[:, columns]


def _reduce_spans(frame, keys):
    return frame.groupby(keys, sort=False, as_index=False).agg(
        allele_frequency=("allele_frequency", "first"),
        inferred_span=("inferred_span", "first"), true_span=("true_span", "min"),
    )


def _reduce_ancestor_chunk(task):
    frame = task.frame.loc[task.frame.version.isin(task.config.versions)].copy()
    # Inferred time is the ancestor's focal-frequency proxy. Raw focal DAFs
    # can disagree within a mispolarised ancestor; this is one value per ancestor.
    frame = frame.rename(columns={"inferred_time": "allele_frequency"})
    keys = list(task.config.ancestor_keys)
    return AncestorFrames(_reduce_overshoots(frame, keys), _reduce_spans(frame, keys))


def _true_focal_frequencies(data_folder, data_prefix):
    """Unflipped focal counts/frequencies from each error-free variant store."""
    folder = pathlib.Path(data_folder).expanduser() / "zarr_vcfs"
    stores = _variant_stores(folder, data_prefix)
    frames = []
    for store in stores:
        if store.geno_multiplier != 0 or store.phase_ser != 0 or store.mispol_rate != 0:
            continue
        group = zarr.open_group(store.path, mode="r")
        counts = group["variant_allele_count"][:]
        total_alleles = group["call_genotype"].shape[1] * group["call_genotype"].shape[2]
        frames.append(pd.DataFrame({
            "replicate": store.replicate,
            "focal_position": group["variant_position"][:],
            "true_derived_count": counts,
            "true_allele_frequency": counts / total_alleles,
        }))
    return pd.concat(frames, ignore_index=True)


def _ancestor_tasks(reader, truth, config):
    for frame in reader:
        frame = frame.merge(truth, on=["replicate", "focal_position"], how="left",
                            validate="many_to_one")
        if frame.true_derived_count.isna().any():
            raise ValueError("Focal positions are missing from the error-free variant stores")
        yield AncestorTask(frame, config)


def preprocess_data(data_folder, data_prefix, config=None):
    """Write ancestor-level ``-overshoot.csv`` and ``-spans.csv`` beside the input.

    Each inferred ancestor is identified within its replicate, version and
    numeric error profile. If any focal association is mispolarised, both sides
    use only mispolarised associations. Otherwise all associations are eligible.
    Overshoot is the signed minimum within that set, independently on each side.
    True focal counts/frequencies come from error-free variant stores. True span
    is the minimum over all associations; inferred span is invariant per ancestor.
    ``load_data`` and every ancestor plot consume only these reduced tables.
    """
    if config is None:
        config = PlotConfig()
    folder = pathlib.Path(data_folder).expanduser() / "dataframes"
    paths = AncestorPaths(folder / f"{data_prefix}-overshoot.csv",
                          folder / f"{data_prefix}-spans.csv")
    columns = list(config.ancestor_keys)
    columns += ["side", "inferred_time", "inferred_span", "true_span", "overshoot",
                "true_focal_site", "focal_position", "focal_site_mispolarised"]
    reader = pd.read_csv(folder / f"{data_prefix}-aggregated.csv", usecols=columns,
                         dtype={"version": str}, chunksize=config.preprocess_chunk_rows)
    truth = _true_focal_frequencies(data_folder, data_prefix)
    tasks = _ancestor_tasks(reader, truth, config)
    workers = config.workers
    if workers is None:
        workers = multiprocessing.cpu_count()
    if workers == 1:
        results = list(map(_reduce_ancestor_chunk, tasks))
    else:
        context = multiprocessing.get_context("spawn")
        with context.Pool(processes=workers) as pool:
            results = list(pool.imap(_reduce_ancestor_chunk, tasks))
    overshoots = pd.concat([result.overshoots for result in results], ignore_index=True)
    spans = pd.concat([result.spans for result in results], ignore_index=True)
    # Ancestor associations can straddle chunk boundaries; apply the same
    # reduction again after collecting independently processed chunks.
    keys = list(config.ancestor_keys)
    overshoots = _reduce_overshoots(overshoots, keys)
    spans = _reduce_spans(spans, keys)
    overshoots.to_csv(paths.overshoots, index=False)
    spans.to_csv(paths.spans, index=False)
    logger.info("Wrote %s ancestor sides and %s ancestor spans", len(overshoots), len(spans))
    return paths


def load_data(data_folder, data_prefix, config=None):
    """Load the reduced tables from ``preprocess_data`` and discover variant stores."""
    if config is None:
        config = PlotConfig()
    folder = pathlib.Path(data_folder).expanduser()
    overshoots = pd.read_csv(folder / "dataframes" / f"{data_prefix}-overshoot.csv")
    spans = pd.read_csv(folder / "dataframes" / f"{data_prefix}-spans.csv")
    overshoots = overshoots.loc[overshoots.version.isin(config.versions)].copy()
    spans = spans.loc[spans.version.isin(config.versions)].copy()
    if set(spans.version) != set(config.versions):
        raise ValueError(f"Ancestor CSV must contain versions {config.versions}")
    stores = _variant_stores(folder / "zarr_vcfs", data_prefix)
    return ChapterData(overshoots, spans, stores, config)


def save_figure(fig, figure_folder, name, config):
    """Save a 300-dpi PNG by default, display it in the notebook, and close it."""
    folder = pathlib.Path(figure_folder).expanduser()
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.png"
    fig.savefig(path, dpi=config.dpi, bbox_inches="tight", facecolor="white")
    logger.info("Saved %s", path)
    plt.show()
    plt.close(fig)


def _no_errors(frame):
    return frame.loc[
        (frame.geno_multiplier == 0)
        & (frame.phase_ser == 0)
        & (frame.mispol_rate == 0)
    ].copy()


def _frequency_bin_edges(frequencies, num_bins=5, cutoffs=None):
    """One-significant-digit quantiles over all DAFs, or explicit bin edges.

    The outer bounds round outwards to include every observation. Tied rounded
    quantiles merge; ``cutoffs`` overrides both rounding and ``num_bins``.
    """
    if cutoffs is not None:
        edges = np.asarray(cutoffs, dtype=float)
        if edges.ndim != 1 or len(edges) < 2:
            raise ValueError("cutoffs must contain at least two edges")
        if not np.all(np.isfinite(edges)) or np.any(np.diff(edges) <= 0):
            raise ValueError("cutoffs must be finite and strictly increasing")
        return edges
    if num_bins < 1:
        raise ValueError("num_bins must be at least 1")
    frequencies = np.asarray(frequencies)
    boundaries = np.quantile(frequencies, np.linspace(0, 1, num_bins + 1))
    rounded = boundaries.copy()
    positive = boundaries > 0
    magnitudes = 10.0 ** np.floor(np.log10(boundaries[positive]))
    rounded[positive] = np.ceil(boundaries[positive] / magnitudes) * magnitudes
    if boundaries[0] > 0:
        magnitude = 10.0 ** np.floor(np.log10(boundaries[0]))
        rounded[0] = np.floor(boundaries[0] / magnitude) * magnitude
    edges = np.unique(rounded)
    if len(edges) < 2:
        raise ValueError("Quantile binning requires distinct frequency bounds")
    if len(edges) - 1 != num_bins:
        logger.warning("Rounded/tied quantiles yield %s of %s requested bins",
                       len(edges) - 1, num_bins)
    return edges


def _ancestor_boxplot(data, figure_folder, variable, num_bins, cutoffs, conf=None, alpha=None):
    config = data.config
    spans = _no_errors(data.spans)
    true_spans = spans.drop_duplicates(["replicate", "inferred_node"])
    edges = _frequency_bin_edges(true_spans.allele_frequency, num_bins, cutoffs)
    if variable == "overshoot":
        if not 0 < conf < 0.5:
            raise ValueError("conf must be between 0 and 0.5")
        if alpha is None:
            alpha = config.jitter_alpha
        rng = np.random.default_rng(config.jitter_seed)
    num_bins = len(edges) - 1
    cut_edges = edges.copy()
    cut_edges[-1] = np.nextafter(edges[-1], np.inf)
    labels = [f"{left:g}–{right:g}" for left, right in zip(edges[:-1], edges[1:])]
    fig, axes = plt.subplots(
        2, 1, figsize=config.box_size, sharex=True,
        gridspec_kw={"height_ratios": [4, 1], "hspace": 0.08},
    )
    groups = []
    if variable == "span":
        groups.append(("True", config.true_color, true_spans, "true_span"))
        for version, color in zip(config.versions, config.version_colors):
            groups.append((f"tsinfer {version}", color,
                           spans.loc[spans.version == version], "inferred_span"))
    else:
        frame = _no_errors(data.overshoots)
        for version, color in zip(config.versions, config.version_colors):
            groups.append((f"tsinfer {version}", color,
                           frame.loc[frame.version == version], "overshoot"))
    width = 0.8 / len(groups)
    handles = []
    for index, (label, color, subset, column) in enumerate(groups):
        bin_ids = pd.cut(subset.allele_frequency, cut_edges, labels=False, right=False)
        values = []
        positions = []
        for bin_index in range(num_bins):
            selected = subset.loc[bin_ids == bin_index, column].to_numpy()
            if len(selected) > 0:
                values.append(selected / config.bp_per_mb)
                positions.append(bin_index - 0.4 + width * (index + 0.5))
        if variable == "span":
            axes[0].boxplot(
                values, positions=positions, widths=width * 0.85,
                patch_artist=True, manage_ticks=False, showfliers=False,
                boxprops={"facecolor": color}, medianprops={"color": "black"},
            )
        else:
            for selected, position in zip(values, positions):
                _draw_overshoot_group(axes[0], selected, position, width * 0.85,
                                      color, rng, config, conf, alpha)
        if variable == "span":
            handles.append(mpatches.Patch(color=color, label=label))
        if index == 0:
            ancestor_rows = subset.drop_duplicates(["replicate", "inferred_node"])
            ancestor_bins = pd.cut(ancestor_rows.allele_frequency, cut_edges,
                                   labels=False, right=False)
            counts = ancestor_bins.value_counts().reindex(range(num_bins), fill_value=0)
            axes[1].bar(np.arange(num_bins), counts, width=width * 0.85 * 2,
                        facecolor="white", edgecolor="black")
    if variable == "span":
        axes[0].legend(handles=handles, frameon=False, loc="lower center",
                       bbox_to_anchor=(0.5, 1.01), ncol=len(groups))
    else:
        fig.set_layout_engine("constrained")
        _overshoot_legends(fig, config, conf, alpha)
    if variable == "span":
        axes[0].set_yscale("log")
        axes[0].set_ylabel("Ancestor span (Mb)")
    else:
        _overshoot_zero_line(axes[0])
        axes[0].set_ylabel("Overshoot (Mb)")
    axes[1].set_ylabel("Ancestors")
    axes[1].yaxis.set_major_locator(mticker.MaxNLocator(nbins=2, integer=True))
    axes[1].yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    axes[1].yaxis.set_minor_formatter(mticker.NullFormatter())
    axes[1].set_xticks(np.arange(num_bins), labels, rotation=25)
    axes[1].set_xlabel("Inferred ancestor frequency")
    axes[0].set_xlim(-0.6, num_bins - 0.4)
    if variable == "span":
        fig.subplots_adjust(bottom=0.18, left=0.1, right=0.98, top=0.9)
    return save_figure(fig, figure_folder, f"ancestor_{variable}_boxplot", config)


def plot_ancestor_spans(data, figure_folder, num_bins=5, cutoffs=None):
    """One inferred span and minimum true span per ancestor in DAF bins."""
    return _ancestor_boxplot(data, figure_folder, "span", num_bins, cutoffs)


def plot_ancestor_overshoot(data, figure_folder, num_bins=5, cutoffs=None, conf=0.01, alpha=None):
    """Minimum left/right overshoots per ancestor, using the span figure DAF bins."""
    return _ancestor_boxplot(data, figure_folder, "overshoot", num_bins, cutoffs, conf, alpha)


def _genotype_counts(task):
    truth = zarr.open_group(task.truth.path, mode="r")
    observed = zarr.open_group(task.observed.path, mode="r")
    true_calls = truth["call_genotype"]
    observed_calls = observed["call_genotype"]
    if true_calls.shape != observed_calls.shape:
        raise ValueError(f"Genotype shapes differ for {task.observed.path}")
    if not np.array_equal(truth["sample_id"][:], observed["sample_id"][:]):
        raise ValueError(f"Sample order differs for {task.observed.path}")
    edges = np.linspace(0, 1, task.bins + 1)
    denominators = np.zeros((2, task.bins), dtype=np.int64)
    numerators = np.zeros_like(denominators)
    changed = 0
    total = 0
    for start in range(0, true_calls.shape[0], true_calls.chunks[0]):
        stop = min(start + true_calls.chunks[0], true_calls.shape[0])
        region = slice(start, stop)
        if not np.array_equal(truth["variant_position"][region],
                              observed["variant_position"][region]):
            raise ValueError(f"Variant order differs for {task.observed.path}")
        true_genotypes = true_calls[region]
        observed_genotypes = observed_calls[region]
        true_dosage = true_genotypes.sum(axis=2)
        observed_dosage = observed_genotypes.sum(axis=2)
        daf = true_dosage.mean(axis=1) / true_calls.shape[2]
        changed += np.count_nonzero(true_dosage != observed_dosage)
        total += true_dosage.size
        for index, dosage in enumerate((0, 2)):
            homozygous = true_dosage == dosage
            to_het = homozygous & (observed_dosage == 1)
            site_denominators = homozygous.sum(axis=1)
            site_numerators = to_het.sum(axis=1)
            denominators[index] += np.histogram(daf, bins=edges, weights=site_denominators)[0]
            numerators[index] += np.histogram(daf, bins=edges, weights=site_numerators)[0]
    return GenotypeCounts(task.observed.geno_multiplier, changed, total,
                          denominators, numerators)


def summarise_genotype_errors(data):
    """Compare diploid dosage against truth, streaming one variant chunk at a time.

    A genotype error changes dosage (00, 01/10, or 11), so a phase reversal is
    never counted as a genotype error. Conditional rates use the number of
    original homozygous calls in each DAF bin as their denominator. Replicates
    are pooled by counts, not by averaging rates. Worker tasks use the process
    pool's queues and run independently for each replicate and multiplier.
    """
    config = data.config
    replicates = sorted(data.spans.replicate.unique())
    tasks = []
    for replicate in replicates:
        truth = data.store(replicate)
        for multiplier in config.genotype_multipliers:
            observed = data.store(replicate, geno_multiplier=multiplier)
            tasks.append(GenotypeTask(truth, observed, config.transition_bins))
    workers = config.workers
    if workers is None:
        workers = min(multiprocessing.cpu_count(), len(tasks))
    if workers == 1:
        results = list(map(_genotype_counts, tasks))
    else:
        context = multiprocessing.get_context("spawn")
        with context.Pool(processes=workers) as pool:
            results = list(pool.imap(_genotype_counts, tasks))
    rate_rows = []
    transition_rows = []
    edges = np.linspace(0, 1, config.transition_bins + 1)
    for multiplier in config.genotype_multipliers:
        selected = [result for result in results if result.multiplier == multiplier]
        changed = sum(result.changed for result in selected)
        total = sum(result.total for result in selected)
        denominators = np.sum([result.homozygous for result in selected], axis=0)
        numerators = np.sum([result.to_heterozygous for result in selected], axis=0)
        rate_rows.append({"multiplier": multiplier, "changed": changed,
                          "total": total, "error_rate": changed / total})
        for index, transition in enumerate(("00 → 01/10", "11 → 01/10")):
            for bin_index in range(config.transition_bins):
                denominator = denominators[index, bin_index]
                numerator = numerators[index, bin_index]
                rate = numerator / denominator if denominator > 0 else np.nan
                transition_rows.append({
                    "multiplier": multiplier, "transition": transition,
                    "daf": (edges[bin_index] + edges[bin_index + 1]) / 2,
                    "changed": numerator, "total": denominator, "error_rate": rate,
                })
    return GenotypeSummary(pd.DataFrame(rate_rows), pd.DataFrame(transition_rows))


def plot_genotype_error_scaling(summary, figure_folder, config=None):
    """Observed diploid genotype error rate versus linear scaling from 1×."""
    if config is None:
        config = PlotConfig()
    rates = summary.rates.sort_values("multiplier")
    base = rates.loc[rates.multiplier == 1, "error_rate"].item()
    fig, ax = plt.subplots(figsize=config.curve_size, layout="constrained")
    colors = plt.get_cmap(config.genotype_scaling_cmap)(config.genotype_color_positions)
    ax.plot(rates.multiplier, rates.error_rate, "o-", color=colors[0],
            label="Observed")
    linear_x = np.geomspace(rates.multiplier.min(), rates.multiplier.max(), config.curve_points)
    linear_y = base * linear_x
    ax.plot(linear_x, linear_y, "--",
            color=colors[1], label="Linear scaling from 1×")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(rates.multiplier, [f"{value:g}×" for value in rates.multiplier])
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda value, position: f"{value * 100:.2g}%"))
    ax.set_xlabel("Genotype-error multiplier")
    ax.set_ylabel("Diploid genotype error rate")
    ax.legend(frameon=False)
    return save_figure(fig, figure_folder, "genotype_error_scaling", config)


def plot_genotype_error_transitions(summary, figure_folder, config=None):
    """Conditional homozygote-to-heterozygote rates by true DAF at 1× and 10×."""
    if config is None:
        config = PlotConfig()
    fig, axes = plt.subplots(1, len(config.transition_multipliers),
                             figsize=config.transition_size, sharey=True,
                             squeeze=False, layout="constrained")
    colors = plt.get_cmap(config.genotype_transition_cmap)(config.genotype_color_positions)
    for ax, multiplier in zip(axes[0], config.transition_multipliers):
        subset = summary.transitions.loc[summary.transitions.multiplier == multiplier]
        for (label, group), color in zip(subset.groupby("transition", sort=False),
                                         colors):
            positive_rates = group.error_rate.where(group.error_rate > 0)
            ax.plot(group.daf, positive_rates, color=color, label=label)
        ax.set_xlim(0, 1)
        ax.set_yscale("log")
        ax.set_xlabel("True derived allele frequency")
        ax.text(0.04, 0.96, f"{multiplier:g}×", transform=ax.transAxes, va="top")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda value, position: f"{value * 100:.2g}%"))
    selected = summary.transitions.loc[
        summary.transitions.multiplier.isin(config.transition_multipliers)
    ]
    positive_rates = selected.loc[selected.error_rate > 0, "error_rate"]
    axes[0, 0].set_ylim(positive_rates.min() / 1.2, positive_rates.max() * 1.2)
    axes[0, 0].set_ylabel("Homozygote-to-heterozygote error rate")
    axes[0, -1].legend(frameon=False)
    return save_figure(fig, figure_folder, "genotype_error_transitions", config)


def _overshoot_colors(rates, cmap_name):
    colors = plt.get_cmap(cmap_name)(np.linspace(0.25, 0.8, len(rates)))
    return dict(zip(rates, colors))


def _overshoot_histogram(data, frame, by, rates, legend_title,
                         name, figure_folder, reverse_order=False):
    config = data.config
    frame = frame.loc[frame[by].isin(rates)].copy()
    values = frame.overshoot.to_numpy() / config.bp_per_mb
    # Shared edges preserve directly comparable counts between all profiles.
    bin_edges = np.histogram_bin_edges(values, bins=config.histogram_bins)
    counts = {}
    max_count = 0
    for version in config.versions:
        for rate in rates:
            selected = frame.loc[(frame.version == version) & (frame[by] == rate)]
            if selected.empty:
                raise ValueError(f"No overshoots for {version}, {by}={rate}")
            histogram = np.histogram(selected.overshoot / config.bp_per_mb, bin_edges)[0]
            counts[version, rate] = histogram
            max_count = max(max_count, histogram.max())
    fig, axes = plt.subplots(2, 1, figsize=config.histogram_size, sharex=True,
                             gridspec_kw={"hspace": 0})
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    for row, (ax, version, cmap_name) in enumerate(zip(axes, config.versions,
                                                      config.overshoot_cmaps)):
        palette = _overshoot_colors(rates, cmap_name)
        for rate in reversed(rates):
            histogram = counts[version, rate].copy()
            if reverse_order:
                histogram[centers >= 0] = 0
            ax.bar(bin_edges[:-1], histogram, width=np.diff(bin_edges), align="edge",
                   color=palette[rate], linewidth=0, bottom=0)
        if reverse_order:
            for rate in rates:
                histogram = counts[version, rate].copy()
                histogram[centers < 0] = 0
                ax.bar(bin_edges[:-1], histogram, width=np.diff(bin_edges), align="edge",
                       color=palette[rate], linewidth=0, bottom=0)
        ax.set_yscale("log")
        ax.set_ylim(0.8, max_count * 1.2)
        max_power = int(np.floor(np.log10(max_count)))
        ax.set_yticks([10 ** power for power in range(1, max_power + 1)])
        ax.axvline(0, color=config.zero_color, linestyle="--", linewidth=1)
        ax.set_ylabel("Ancestor sides")
        if row == 1:
            ax.invert_yaxis()
        handles = []
        for rate in rates:
            label = f"{rate:g}×" if by == "geno_multiplier" else f"{rate:g}"
            handles.append(mpatches.Patch(color=palette[rate], label=label))
        ax.legend(handles=handles, title=legend_title,
                  loc="upper right" if row == 0 else "lower right")
        ax.text(0.02, 0.96 if row == 0 else 0.04, f"tsinfer {version}",
                transform=ax.transAxes, va="top" if row == 0 else "bottom",
                bbox={"facecolor": "white", "edgecolor": "#cccccc", "pad": 4})
    axes[1].set_xlim(bin_edges[0], bin_edges[-1])
    axes[1].set_xlabel("Overshoot (Mb)")
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.14, top=0.98)
    return save_figure(fig, figure_folder, f"{name}_hist", config)


def _draw_overshoot_group(ax, values, position, box_width, color, rng, config, conf, alpha):
    """White boxes, sampled central/all tail dots with one opacity, quantile caps."""
    ax.boxplot(
        [values], positions=[position], widths=box_width,
        patch_artist=True, manage_ticks=False, showfliers=False,
        boxprops={"facecolor": "white", "edgecolor": "black"},
        whiskerprops={"color": "black"}, capprops={"color": "black"},
        medianprops={"color": "black", "linewidth": 1.2}, zorder=3,
    )
    low, high = np.quantile(values, [conf, 1 - conf])
    outliers = (values < low) | (values > high)
    central = values[~outliers]
    sample_size = min(len(central), config.jitter_max_points)
    sampled = rng.choice(central, size=sample_size, replace=False)
    sampled = np.concatenate([sampled, values[outliers]])
    jitter = rng.uniform(-config.jitter_width / 2, config.jitter_width / 2, len(sampled))
    ax.scatter(position + jitter, sampled, s=config.jitter_size, alpha=alpha,
               color=color, edgecolors="none", rasterized=True, zorder=2)
    ax.hlines([low, high], position - config.jitter_width / 2,
              position + config.jitter_width / 2, color="black",
              linestyle=config.quantile_line_style, linewidth=config.quantile_line_width,
              zorder=5)


def _overshoot_zero_line(ax):
    """Match the border, above jitter (2) and below boxes (3)."""
    ax.axhline(0, color="black", linestyle="-",
               linewidth=ax.spines["left"].get_linewidth(), zorder=2.5)


def _overshoot_legends(fig, config, conf, alpha):
    handles = [
        mlines.Line2D([], [], marker="o", linestyle="none", color=color,
                      alpha=alpha, markeredgecolor="none", label=f"tsinfer {version}")
        for version, color in zip(config.versions, config.version_colors)
    ]
    handles.append(mlines.Line2D([], [], color="black", linestyle=config.quantile_line_style,
                                 label=f"{conf * 100:g}% tail quantiles"))
    fig.legend(handles=handles, frameon=False, loc="outside right center")


def _overshoot_boxplot(data, frame, by, rates, label, name, figure_folder, conf, alpha):
    """Grouped boxes of reduced ancestor sides; see `_draw_overshoot_group`."""
    if not 0 < conf < 0.5:
        raise ValueError("conf must be between 0 and 0.5")
    config = data.config
    if alpha is None:
        alpha = config.jitter_alpha
    rng = np.random.default_rng(config.jitter_seed)
    fig, ax = plt.subplots(figsize=config.overshoot_box_size, layout="constrained")
    group_width = 0.8 / len(config.versions)
    box_width = config.jitter_width * config.overshoot_box_width_ratio
    for version_index, (version, color) in enumerate(zip(config.versions, config.version_colors)):
        for index, rate in enumerate(rates):
            subset = frame.loc[(frame.version == version) & (frame[by] == rate)]
            if subset.empty:
                raise ValueError(f"No overshoots for {version}, {by}={rate}")
            values = subset.overshoot.to_numpy() / config.bp_per_mb
            position = index - 0.4 + group_width * (version_index + 0.5)
            _draw_overshoot_group(ax, values, position, box_width, color, rng, config, conf, alpha)
    _overshoot_zero_line(ax)
    ax.set_ylabel("Overshoot (Mb)")
    ax.set_xlabel(label)
    labels = [f"{rate:g}×" if by == "geno_multiplier" else f"{rate:g}" for rate in rates]
    ax.set_xticks(range(len(rates)), labels)
    ax.set_xlim(-0.5, len(rates) - 0.5)
    ax.grid(axis="y", alpha=0.15)
    _overshoot_legends(fig, config, conf, alpha)
    return save_figure(fig, figure_folder, f"{name}_box", config)


@dataclasses.dataclass
class TailSteps:
    edges: np.ndarray
    fractions: np.ndarray


def _tail_steps(magnitudes, num_ancestors):
    """Strict exceedance on intervals between observed positive magnitudes.

    The denominator includes every ancestor, including those with no overshoot
    of this sign. The zero fraction beyond the largest value is not drawn on
    the log axis and is not replaced with an artificial nonzero value.
    """
    positive = magnitudes[magnitudes > 0]
    positions, counts = np.unique(positive, return_counts=True)
    if len(positions) == 0:
        return TailSteps(np.array([0]), np.array([]))
    counts_below = np.concatenate(([0], np.cumsum(counts)[:-1]))
    fractions = (len(positive) - counts_below) / num_ancestors
    edges = np.concatenate(([0], positions))
    return TailSteps(edges, fractions)


def _overshoot_tails(data, frame, by, rates, label, name, figure_folder):
    """Fraction of ancestors whose retained left or right side exceeds a magnitude.

    Panels show negative and positive overshoot separately. Each ancestor is
    counted once in each panel, even when both retained sides exceed the bound.
    """
    config = data.config
    fig, axes = plt.subplots(1, 2, figsize=config.overshoot_tail_size,
                             sharex=True, sharey=True, layout="constrained")
    handles = []
    max_magnitude = 0
    min_fraction = 1
    keys = ["replicate", "inferred_node"]
    for version, cmap_name, line_style in zip(config.versions, config.overshoot_cmaps,
                                             config.version_line_styles):
        palette = _overshoot_colors(rates, cmap_name)
        for rate in rates:
            subset = frame.loc[(frame.version == version) & (frame[by] == rate)]
            if subset.empty:
                raise ValueError(f"No overshoots for {version}, {by}={rate}")
            ancestors = subset.groupby(keys).overshoot.agg(["min", "max"])
            num_ancestors = len(ancestors)
            min_fraction = min(min_fraction, 1 / num_ancestors)
            magnitudes = [-ancestors["min"].to_numpy(), ancestors["max"].to_numpy()]
            for ax, values in zip(axes, magnitudes):
                steps = _tail_steps(values / config.bp_per_mb, num_ancestors)
                if len(steps.fractions) > 0:
                    ax.stairs(steps.fractions, steps.edges, baseline=None,
                              color=palette[rate], linestyle=line_style)
                    max_magnitude = max(max_magnitude, steps.edges[-1])
            category = f"{rate:g}×" if by == "geno_multiplier" else f"{rate:g}"
            handles.append(mlines.Line2D([], [], color=palette[rate], linestyle=line_style,
                                         label=f"{version}: {category}"))
    for ax, sign in zip(axes, ("Negative", "Positive")):
        ax.set_xlabel(f"{sign} overshoot magnitude (Mb)")
        ax.set_yscale("log")
        ax.set_xlim(0, max_magnitude)
        ax.set_ylim(min_fraction * 0.8, 1)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda value, position: f"{value * 100:g}%"
        ))
        ax.grid(axis="y", alpha=0.15)
    axes[0].set_ylabel("Fraction of ancestors beyond magnitude")
    fig.legend(handles=handles, title=label, loc="outside right center", frameon=False)
    return save_figure(fig, figure_folder, f"{name}_tails", config)


def _genotype_overshoots(data):
    frame = data.overshoots
    return frame.loc[(frame.phase_ser == 0) & (frame.mispol_rate == 0)]


def plot_genotype_overshoot_hist(data, figure_folder):
    """Mirrored histograms for pure genotype-error profiles."""
    return _overshoot_histogram(
        data, _genotype_overshoots(data), "geno_multiplier", data.config.genotype_multipliers,
        "Genotype-error multiplier", "genotype_error_overshoot", figure_folder,
        reverse_order=True,
    )


def plot_genotype_overshoot_box(data, figure_folder, conf=0.01, alpha=None):
    """Grouped boxplots for pure genotype-error profiles; see `_overshoot_boxplot`."""
    return _overshoot_boxplot(
        data, _genotype_overshoots(data), "geno_multiplier", data.config.genotype_multipliers,
        "Genotype-error multiplier", "genotype_error_overshoot", figure_folder, conf, alpha,
    )


def plot_genotype_overshoot_tails(data, figure_folder):
    """Signed exceedance fractions by genotype-error multiplier; see `_overshoot_tails`."""
    return _overshoot_tails(
        data, _genotype_overshoots(data), "geno_multiplier", data.config.genotype_multipliers,
        "Genotype-error multiplier", "genotype_error_overshoot", figure_folder,
    )


def plot_mispolarisation_daf(data, figure_folder):
    """Overlay smoothed DAF densities after flipping mispolarised sites.

    The stored allele counts precede mispolarisation. The mask is True for
    correctly polarised sites; its complement selects the DAFs to reverse.
    Fixed-width histograms and Gaussian smoothing avoid KDE evaluation costs
    growing with the number of sites. Each profile integrates to one.
    """
    config = data.config
    replicates = sorted(data.spans.replicate.unique())
    edges = np.linspace(0, 1, config.daf_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    colors = plt.get_cmap("viridis")(np.linspace(0.1, 0.85, len(config.mispol_rates)))
    fig, ax = plt.subplots(figsize=config.curve_size, layout="constrained")
    for rate, color in zip(config.mispol_rates, colors):
        counts = np.zeros(config.daf_bins, dtype=np.int64)
        for replicate in replicates:
            store = data.store(replicate, mispol_rate=rate)
            group = zarr.open_group(store.path, mode="r")
            total_alleles = group["call_genotype"].shape[1] * group["call_genotype"].shape[2]
            daf = group["variant_allele_count"][:] / total_alleles
            mispolarised = ~group["variant_mispolarisation_mask"][:].astype(bool)
            daf[mispolarised] = 1 - daf[mispolarised]
            counts += np.histogram(daf, edges)[0]
        density = counts / counts.sum() / np.diff(edges)
        density = ndi.gaussian_filter1d(density, config.daf_smoothing_bins, mode="reflect")
        ax.plot(centers, density, color=color, label=f"{rate:g}")
    ax.set_xlim(0, 1)
    ax.set_yscale("log")
    ax.set_xlabel("Derived allele frequency")
    ax.set_ylabel("Density")
    ax.legend(title="Mispolarisation rate", frameon=False)
    return save_figure(fig, figure_folder, "mispolarisation_daf", config)


def _mispolarisation_overshoots(data, frequency):
    """Select pure profiles using unflipped true focal DAF, including 0.5 in high."""
    frame = data.overshoots
    pure = frame.loc[(frame.geno_multiplier == 0) & (frame.phase_ser == 0)]
    if frequency == "high":
        return pure.loc[pure.true_allele_frequency >= 0.5]
    if frequency == "low":
        return pure.loc[pure.true_allele_frequency < 0.5]
    if frequency == "all":
        return pure
    raise ValueError('frequency must be "high", "low" or "all"')


def plot_mispolarisation_overshoot_hist(data, figure_folder, frequency="all"):
    """Mirrored histograms at high (≥ 0.5), low (< 0.5), or all true DAFs."""
    return _overshoot_histogram(
        data, _mispolarisation_overshoots(data, frequency), "mispol_rate",
        data.config.mispol_rates, "Mispolarisation rate",
        f"mispolarisation_overshoot_{frequency}_frequency", figure_folder,
    )


def plot_mispolarisation_overshoot_box(data, figure_folder, frequency="all", conf=0.01, alpha=None):
    """Grouped mispolarisation boxplots; see `_overshoot_boxplot`."""
    return _overshoot_boxplot(
        data, _mispolarisation_overshoots(data, frequency), "mispol_rate",
        data.config.mispol_rates, "Mispolarisation rate",
        f"mispolarisation_overshoot_{frequency}_frequency", figure_folder, conf, alpha,
    )


def plot_mispolarisation_overshoot_tails(data, figure_folder, frequency="all"):
    """Signed exceedance fractions by mispolarisation rate; see `_overshoot_tails`."""
    return _overshoot_tails(
        data, _mispolarisation_overshoots(data, frequency), "mispol_rate",
        data.config.mispol_rates, "Mispolarisation rate",
        f"mispolarisation_overshoot_{frequency}_frequency", figure_folder,
    )


def plot_mispolarisation_low_count_overshoot_hist(data, figure_folder, max_derived_count=5):
    """Only mispolarised ancestors with true derived count at most the bound."""
    frame = _mispolarisation_overshoots(data, frequency="all")
    selected = frame.loc[
        frame.focal_site_mispolarised & (frame.true_derived_count <= max_derived_count)
    ]
    rates = [rate for rate in data.config.mispol_rates if rate > 0]
    return _overshoot_histogram(
        data, selected, "mispol_rate", rates, "Mispolarisation rate",
        f"mispolarisation_overshoot_derived_count_le_{max_derived_count}", figure_folder,
    )


def _phasing_overshoots(data):
    frame = data.overshoots
    return frame.loc[(frame.geno_multiplier == 0) & (frame.mispol_rate == 0)]


def plot_phasing_overshoot_hist(data, figure_folder):
    """Mirrored histograms for pure phase-switch profiles."""
    return _overshoot_histogram(
        data, _phasing_overshoots(data), "phase_ser", data.config.phase_rates,
        "Switch error rate", "phasing_error_overshoot", figure_folder, reverse_order=True,
    )


def plot_phasing_overshoot_box(data, figure_folder, conf=0.01, alpha=None):
    """Grouped phase-switch boxplots; see `_overshoot_boxplot`."""
    return _overshoot_boxplot(
        data, _phasing_overshoots(data), "phase_ser", data.config.phase_rates,
        "Switch error rate", "phasing_error_overshoot", figure_folder, conf, alpha,
    )


def plot_phasing_overshoot_tails(data, figure_folder):
    """Signed exceedance fractions by switch error rate; see `_overshoot_tails`."""
    return _overshoot_tails(
        data, _phasing_overshoots(data), "phase_ser", data.config.phase_rates,
        "Switch error rate", "phasing_error_overshoot", figure_folder,
    )


def _phase_runs(positions, phase, length):
    """Return physical runs; a switch starts at its saved right-hand variant."""
    changes = np.flatnonzero(phase[1:] != phase[:-1]) + 1
    indices = np.concatenate(([0], changes))
    boundaries = np.concatenate(([0], positions[changes], [length]))
    widths = np.diff(boundaries)
    return boundaries[:-1], widths, phase[indices]


def _draw_phase_pair(ax, lefts, widths, states, y, config):
    for haplotype in range(2):
        for state in (False, True):
            selected = states == state
            runs = np.column_stack((lefts[selected], widths[selected]))
            color = config.phase_colors[haplotype ^ int(state)]
            ax.broken_barh(runs, (y + haplotype * 0.3, 0.24),
                           facecolors=color, edgecolors="none")


def plot_phase_switch_illustration(phasing_folder, data_prefix, figure_folder, config=None):
    """Aligned true/error haplotype pairs for the first individuals on chr1's p arm.

    Colours follow the saved true haplotype identity even at homozygous sites.
    """
    if config is None:
        config = PlotConfig()
    folder = pathlib.Path(phasing_folder).expanduser()
    stores = _variant_stores(folder / "zarr_vcfs", data_prefix)
    profiles = ChapterData(pd.DataFrame(), pd.DataFrame(), stores, config)
    truth = zarr.open_group(profiles.store(0).path, mode="r")
    observed = zarr.open_group(profiles.store(0, phase_ser=config.phase_ser).path, mode="r")
    positions = observed["variant_position"][:]
    if not np.array_equal(positions, truth["variant_position"][:]):
        raise ValueError("Phasing illustration stores have different variant positions")
    if not np.array_equal(truth["sample_id"][:], observed["sample_id"][:]):
        raise ValueError("Phasing illustration stores have different sample order")
    phase = observed["call_genotype_phase"][:].astype(bool)
    length = observed["contig_length"][0]
    individuals = min(config.phase_individuals, phase.shape[1])
    right_mb = min(config.phase_right_mb, length / config.bp_per_mb)
    fig, overview = plt.subplots(1, 2, figsize=config.phase_size, layout="constrained")
    for index in range(individuals):
        pair_lefts, pair_widths, pair_states = _phase_runs(positions, phase[:, index], length)
        y = (individuals - index - 1) * config.phase_row_spacing
        _draw_phase_pair(overview[0], np.array([0]), np.array([length / config.bp_per_mb]),
                         np.array([False]), y, config)
        _draw_phase_pair(overview[1], pair_lefts / config.bp_per_mb,
                         pair_widths / config.bp_per_mb, pair_states, y, config)
    for ax in overview:
        ax.set_xlim(0, right_mb)
        top = (individuals - 1) * config.phase_row_spacing + 0.95
        ax.set_ylim(-0.1, top)
        row_centers = np.arange(individuals) * config.phase_row_spacing + 0.27
        ax.set_yticks(row_centers,
                      [str(index + 1) for index in reversed(range(individuals))])
        ax.set_xlabel("Chromosome 1 p arm (Mb)")
    overview[0].set_ylabel("Individual (two haplotypes each)")
    for ax, label in zip(overview, ("Correct phase", f"SER = {config.phase_ser:g}")):
        ax.text(0.03, 0.97, label, transform=ax.transAxes, va="top")
    handles = [mpatches.Patch(color=color, label=f"True haplotype {index + 1}")
               for index, color in enumerate(config.phase_colors)]
    fig.legend(handles=handles, loc="outside right center", frameon=False)
    return save_figure(fig, figure_folder, "phase_switch_illustration", config)
