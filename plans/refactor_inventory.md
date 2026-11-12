# Refactor inventory (12 November 2026)

Original work is protected in `/private/tmp/dphil_refactor_20261112/pre_refactor.tgz` and `user_changes.patch` (local recovery only; final notebooks are committed). The patch includes the pre-existing pt13, plotting, root-edge and guidance edits. The untracked pt13 plotting test is protected in that external backup before planned deletion. No old tests are retained in this repository.

All `.ipynb` output, attachment and execution-count hashes are in `refactor_notebook_inventory.json`. Experiment file sizes and SHA-256 hashes are in `refactor_experiment_inventory.json`; source definitions and starting hashes are in `refactor_source_inventory.json`. These are a one-time migration inventory, not runtime infrastructure.

## Source destinations

| Original | Owner |
| --- | --- |
| `src/__init__.py` | Remove unused source-root package |
| `src/ch3_anc_perf.py` | Split preparation/analysis into `ch3/nb_utils.py` and plotting into `ch3/nb_plot.py`; final used subset separately into thesis modules |
| `src/ch3_anc_perf_analysis.py` | Remove unused grouping replay and plotting; external pipeline owns these, no current consumer |
| `src/ch3_ancestor_extension.py` | Split preparation/analysis into `ch3/nb_utils.py` and plotting into `ch3/nb_plot.py`; final used subset separately into thesis modules |
| `src/ch3_extract_data.py` | `scripts/extract_zarr_validation_data.py` (standalone) |
| `src/ch3_extract_phase_data.py` | `scripts/extract_phase_data.py` (standalone) |
| `src/ch3_plot.py` | Split preparation/analysis into `ch3/nb_utils.py` and plotting into `ch3/nb_plot.py`; final used subset separately into thesis modules |
| `src/ch5_analysis.py` | Split all definitions by responsibility into `ch5/nb_utils.py` and `ch5/nb_plot.py`; explicitly rename colliding definitions |
| `src/ch5_arg_metrics.py` | Split all definitions by responsibility into `ch5/nb_utils.py` and `ch5/nb_plot.py`; explicitly rename colliding definitions |
| `src/ch5_conditioning.py` | Split all definitions by responsibility into `ch5/nb_utils.py` and `ch5/nb_plot.py`; explicitly rename colliding definitions |
| `src/ch5_conditioning_plots.py` | Split all definitions by responsibility into `ch5/nb_utils.py` and `ch5/nb_plot.py`; explicitly rename colliding definitions |
| `src/ch5_dating.py` | Split all definitions by responsibility into `ch5/nb_utils.py` and `ch5/nb_plot.py`; explicitly rename colliding definitions |
| `src/ch5_dating_plots.py` | Split all definitions by responsibility into `ch5/nb_utils.py` and `ch5/nb_plot.py`; explicitly rename colliding definitions |
| `src/ch5_gap_parents.py` | Split all definitions by responsibility into `ch5/nb_utils.py` and `ch5/nb_plot.py`; explicitly rename colliding definitions |
| `src/ch5_gapfill.py` | Split all definitions by responsibility into `ch5/nb_utils.py` and `ch5/nb_plot.py`; explicitly rename colliding definitions |
| `src/ch5_gapfill_plots.py` | Split all definitions by responsibility into `ch5/nb_utils.py` and `ch5/nb_plot.py`; explicitly rename colliding definitions |
| `src/ch5_hyperparams.py` | Split all definitions by responsibility into `ch5/nb_utils.py` and `ch5/nb_plot.py`; explicitly rename colliding definitions |
| `src/ch5_min_sizes.py` | Split all definitions by responsibility into `ch5/nb_utils.py` and `ch5/nb_plot.py`; explicitly rename colliding definitions |
| `src/ch5_paramsearch.py` | Split all definitions by responsibility into `ch5/nb_utils.py` and `ch5/nb_plot.py`; explicitly rename colliding definitions |
| `src/ch5_perf.py` | Split all definitions by responsibility into `ch5/nb_utils.py` and `ch5/nb_plot.py`; explicitly rename colliding definitions |
| `src/ch5_perf_pt5.py` | Split all definitions by responsibility into `ch5/nb_utils.py` and `ch5/nb_plot.py`; explicitly rename colliding definitions |
| `src/ch5_perf_scaling.py` | Split all definitions by responsibility into `ch5/nb_utils.py` and `ch5/nb_plot.py`; explicitly rename colliding definitions |
| `src/ch5_perf_scaling_plots.py` | Split all definitions by responsibility into `ch5/nb_utils.py` and `ch5/nb_plot.py`; explicitly rename colliding definitions |
| `src/ch5_root_edges.py` | Split all definitions by responsibility into `ch5/nb_utils.py` and `ch5/nb_plot.py`; explicitly rename colliding definitions |
| `src/ch5_stitching.py` | Split all definitions by responsibility into `ch5/nb_utils.py` and `ch5/nb_plot.py`; explicitly rename colliding definitions |
| `src/error_estimation.py` | `src/ch4/error_estimation.py`; exploratory notebook calls via `ch4/nb_utils.py` |
| `src/error_validation.py` | `src/ch4/error_validation.py`; exploratory notebook calls via `ch4/nb_utils.py` |

## Notebook pairs

Both sides were read before editing. Divergent source cells will be reconciled individually while retaining output-bearing cells, narratives and all saved outputs.

| Paired source | Source / notebook cells | Initial comparison |
| --- | --- | --- |
| `notebooks/ch3/figure_gen.py` | 135 / 135 | Identical |
| `notebooks/ch4/initial_testing.py` | 57 / 57 | Identical |
| `notebooks/ch5/ancestor_coverage.py` | 24 / 24 | Identical |
| `notebooks/ch5/initial_testing.py` | 13 / 13 | Identical |
| `notebooks/ch5/stitching__scratch.py` | 27 / 27 | Identical |
| `notebooks/ch5/stitching_pt1.py` | 28 / 28 | Identical |
| `notebooks/ch5/stitching_pt10_age_performance_review.py` | 7 / 7 | Identical |
| `notebooks/ch5/stitching_pt11_gapfill.py` | 34 / 34 | Identical |
| `notebooks/ch5/stitching_pt12_conditioning.py` | 16 / 16 | Identical |
| `notebooks/ch5/stitching_pt13_jerome_update.py` | 38 / 38 | Identical |
| `notebooks/ch5/stitching_pt2.py` | 62 / 62 | Identical |
| `notebooks/ch5/stitching_pt3_hyperparams.py` | 38 / 38 | Identical |
| `notebooks/ch5/stitching_pt4_perf.py` | 15 / 15 | Identical |
| `notebooks/ch5/stitching_pt5.py` | 17 / 17 | Identical |
| `notebooks/ch5/stitching_pt6_min_sizes.py` | 37 / 37 | Identical |
| `notebooks/ch5/stitching_pt7_paramsearch.py` | 39 / 39 | Identical |
| `notebooks/ch5/stitching_pt8_hpc_perf1.py` | 32 / 32 | Identical |
| `notebooks/ch5/stitching_pt9_hpc_perf2.py` | 32 / 32 | Identical |
| `notebooks/ch5/tie_breaking.py` | 22 / 22 | Identical |

## Experiment and data destinations

- Both Python drivers → flat `scripts/` with CLI settings; reusable functions → chapter 5 rough modules.
- `experiments/ch5/data/*` → `data/ch5/*`, merging only disjoint files and verifying every hash.
- `sim_n300_10mbp_error_populations.csv` → `data/ch5/` (tracked scientific metadata).
- Analysis YAML → explicit notebook cells and plot-call keyword arguments, then delete. Upstream `stitching_pt3_experiment_config.yaml` is run provenance → `data/ch5/match_eval/`.
- Experiment README → `notebooks/ch5/README.md`; script instructions → `scripts/README.md`.
- `data/HapMapII_GRCh38` and shared genotype stores → `data/general/`; chapter 4 intermediate data → `data/ch4/`; chapter 5 root data → `data/ch5/`. Existing chapter 3/5 data remain owned there; unrelated historical artifacts remain recorded and preserved.

## Old test files (scheduled deletion)

- `tests/.ipynb_checkpoints/test_ch3_anc_perf-checkpoint.py`
- `tests/.ipynb_checkpoints/test_ch3_extract_data-checkpoint.py`
- `tests/.ipynb_checkpoints/test_ch5_gapfill-checkpoint.py`
- `tests/.ipynb_checkpoints/test_ch5_paramsearch-checkpoint.py`
- `tests/test_ch3_anc_perf.py`
- `tests/test_ch3_boxplots.py`
- `tests/test_ch3_combined_errors.py`
- `tests/test_ch3_extract_data.py`
- `tests/test_ch3_extract_spans.py`
- `tests/test_ch3_mispolarisation.py`
- `tests/test_ch3_overlap_errors.py`
- `tests/test_ch3_overshoot_summary.py`
- `tests/test_ch3_phasing.py`
- `tests/test_ch3_span_curves.py`
- `tests/test_ch3_tgp_daf.py`
- `tests/test_ch5_analysis.py`
- `tests/test_ch5_arg_metrics.py`
- `tests/test_ch5_conditioning.py`
- `tests/test_ch5_dating.py`
- `tests/test_ch5_gapfill.py`
- `tests/test_ch5_hyperparams.py`
- `tests/test_ch5_paramsearch.py`
- `tests/test_ch5_perf.py`
- `tests/test_ch5_perf_scaling.py`
- `tests/test_ch5_perf_scaling_plots.py`
- `tests/test_ch5_stitching.py`
- `tests/test_error_validation.py`
- `tests/test_extract_1kgp_perf.py`

## Chapter 3 thesis audit

The existing chapter text was read from `../dphil-thesis/tex/ch3_anc.tex`. Final figures retain its basenames and formats; `anc_eval_pipe.png` is a manually supplied illustration. The toy extension/tree diagrams are generated from explicit parameters. Derived quantities must use identified data; literature quotations and algebraic constants are not generated analysis claims. The lines below identify the figure references and numerical claims to map in phase 6.

```text
20:as their focal frequency increased beyond about 60\%. We suspected that there
34:was ultimately integrated into tsinfer 0.5, so that is the name I refer to it as
252:% above 0.8, I explored a radical approach which forced ancestors above a given
260:% 	\begin{center} \includegraphics[width=\textwidth]{../figures/ch3/root_tree.png}
306:  \includegraphics[width=\textwidth]{../figures/ch3/ancestor_extension_trees.pdf}
314:infinite-sites model, if all possible two-site haplotypes 00, 01, 10, and 11 for
318:supplies 00, $a$ provides 10, and $f$ has 01. We do not reach the same
319:conclusion comparing sites 1 and 2, because the doubleton at site 2 is nested in
386:  \includegraphics[width=\textwidth]{../figures/ch3/ancestor_extension_comparison.png}
388:    extension in tsinfer 0.4 and 0.5. The focal site is 1, with
412:$\boldsymbol{p} = [p_1, p_2, \ldots, p_k, s]$ be the length-$(k+1)$ array of site
471:than 10 Mb as \textit{severe undershoots}.
507:To validate the changes to ancestor inference in tsinfer 0.5, I developed a
517:	\begin{center} \includegraphics[width= 0.8\textwidth]{../figures/ch3/anc_eval_pipe.png}
614:I chose to simulate a 20 Mb region with 6,000 individuals for this evaluation to
622:In most cases at least 75\% of overshoot values are zero or positive, with a median
623:overshoot of 13.8--32.4 kb for tsinfer 0.4 and 10.6--26.1 kb for tsinfer 0.5
635:  \includegraphics[width=\textwidth]{../figures/ch3/demography_overshoot_box.png}
638:    6,000 diploid individuals in a 20 Mb region of chromosome 20.  \textbf{A.}~Medians
640:    percentile intervals (thick lines) and 0.1\%--99.9\% quantile intervals
647:across tsinfer versions and demographies at roughly 50\%. The pooled mismatch
648:fraction shows a negligible relative increase in tsinfer 0.5 of at most 0.024\%.
650:in at least an 86\% higher pooled error fraction than any other simulation. There is no
655:  \includegraphics[width=\textwidth]{../figures/ch3/demography_overlap_errors.png}
668:the chapter, which has 358,285 focal sites carried by 137,123 true and
669:228,445 inferred ancestors.
674:Mean ancestor spans in tsinfer 0.5 are consistently lower than in 0.4,
684:    \includegraphics[width=\textwidth]{../figures/ch3/ancestor_span_frequency_simulation.png}
688:    \includegraphics[width=\textwidth]{../figures/ch3/ancestor_span_frequency_tgp.png}
693:    of 26.0 Mb with 3,202 diploid individuals. Curves show the arithmetic mean
707:  \includegraphics[width=\textwidth]{../figures/ch3/ancestor_overshoot_boxplot.png}
709:    simulated 20 Mb region, binned by observed focal allele frequency. \textbf{A.}~Boxplots with tail 0.1\% quantiles and observations shown. \textbf{B.}~Counts of
718:with 62\% of recent ancestors (DAF < 0.001) having no errors versus only 22\%
720:negligibly in tsinfer 0.5 by at most 0.22\% relative to tsinfer 0.4 in any bin.
724:  \includegraphics[width=\textwidth]{../figures/ch3/ancestor_overlap_errors.png}
750:genotype error rate estimates, accounting for 77\% of the errors in their analysis
794:  \includegraphics[width=0.85\textwidth]{../figures/ch3/genotype_error_scaling.png}
796:    error in the simulated 20 Mb region. A diploid genotype is counted as an
798:    linear scaling extrapolated from the observed $1\times$ rate.}
803:error rate for variants with true minor allele frequency (MAF) > 0.001 is 0.215\%, 572.2\% higher than
804:the estimate of 0.032\% reported by \citeasnoun{masaki2024simultaneous} for UK
806:0.078\% in our results with $\lambda = 1$ and no MAF filter matches the 0.08\%
815:%   \includegraphics[width=\textwidth]{../figures/ch3/genotype_error_transitions.png}
818:%     Panels compare $1\times$ (left) and $10\times$ (right) error multipliers.
826:($\lambda = 0,1,2,5,10$) and summarised the overshoot values in
828:case, the median overshoot is still positive (14.80~kb for tsinfer 0.5 and
829:17.13~kb for tsinfer 0.4) with only 3.3\% of focal-site sides having severe
832:is an undershoot. In the extreme $\lambda = 10$ case, 11.9\% of all focal-site sides
833:have severe undershoots and 75\% have undershoots or zero overshoot.
837:  \includegraphics[width=0.82\textwidth]{../figures/ch3/genotype_error_overshoot_summary.png}
841:    undershoots (more than 10 Mb).}
852:undershoot observations doesn't increase by more than 1.3\% in any bin.
857:  \includegraphics[width=0.82\textwidth]{../figures/ch3/genotype_error_overshoot_diff.png}
880:ranged from 0.6--4.1\%, where the higher probabilities correspond to
882:on the chimpanzee genome, which reported mispolarisation rates of about 0.5\% for
883:typical SNPs and approximately 9.8\% for CpG hotspots \cite{chimpanzee2005initial}.
886:uniformly with constant mispolarisation rates of 1\%, 5\% and 10\% to cover the
892:as extreme as the simulation with a 10\% mispolarisation rate. This is consistent
896:  \includegraphics[width=0.65\textwidth]{../figures/ch3/mispolarisation_daf.png}
898:    simulations of the 20 Mb region of chromosome 20 ($n=12,000$ haplotypes)
899:    with various mispolarisation rates, and the first 20 Mbp of chromosome 20 in
900:    the 1000 Genomes Project ($n=6,404$ haplotypes). Curves were computed with 200 equal-width
917:  \includegraphics[width=\textwidth]{../figures/ch3/mispolarisation_overshoot_correct_focal_diff.png}
943:  \includegraphics[width=\textwidth]{../figures/ch3/mispolarisation_overshoot_flipped_focal_hist.png}
953:Figure~\ref{fig:ch3-mispol-errors}A shows that even a mispolarisation rate of 10\%
960:  \includegraphics[width=\textwidth]{../figures/ch3/mispolarisation_overlap_errors.png}
984:switches of 20.8~Mb in White British UK Biobank trio offspring and 1.9~Mb in
988:rates by targeting mean spacings of 0.2 Mb, 2 Mb and 20 Mb between switch
989:errors, where the 0.2 Mb case is a stress test not based on estimates from a
998:length (20 Mb in this case), $N_{pairs}$ is the total number of consecutive
1013:  \includegraphics[width=\textwidth]{../figures/ch3/phase_switch_illustration.png}
1027:($7.65 \times 10^{-4}$) still results in a positive median overshoot with both
1028:versions, with a median overshoot of 7.66~kb for tsinfer 0.4 and 5.79~kb for
1030:$7.65 \times 10^{-3}$, only 1.63\% of focal-site sides have severe undershoots
1035:  \includegraphics[width=\textwidth]{../figures/ch3/phasing_error_overshoot_summary.png}
1039:    10 Mb).}
1047:0, where there is a maximum increase of 6\% across all bins (Figure~\ref{fig:ch3-phasing-overshoot}B).
1052:  \includegraphics[width=\textwidth]{../figures/ch3/phasing_error_overshoot_diff.png}
1066:2 Mb and finally applied an ancestral allele
1067:mispolarisation probability of 1\%. I will exclude the mispolarised focal sites
1074:threefold increase in pooled mismatch fraction (from 0.046\% to 0.140\%; B),
1079:the maximum overshoot to 7.63 Mb in the presence of error (D).
1084:  \includegraphics[width=\textwidth]{../figures/ch3/combined_errors.png}
1100:and an Out-of-Africa simulation with sample sizes of 100, 1,000, 10,000 and 100,000
1101:haplotypes covering the whole p-arm of chromosome 20 (up to 28.1 Mbp) with
1104:using 196 worker threads and a memory reservation of 950,000 MB.
1111:performance for sample sizes of 100 and 1,000, with only a slight improvement in
1113:$n=100,000$ case, where ancestor matching is 4.2$\times$ faster and sample matching
1114:is $9.7\times$ faster. The 1000 Genomes Project also shows considerable performance
1115:gains, with a $4.7\times$ improvement in sample matching speed, exceeding the $3.3\times$
1116:improvement in the larger simulated $n=10,000$ dataset, with little improvement
1122:  \includegraphics[width=0.8\textwidth]{../figures/ch3/inference_performance_speedup.png}
1124:    tsinfer 0.5 relative to 0.4 in simulated 28.1 Mb regions with 100 to 100,000
1125:    haplotypes, and in the 1KGP chromosome 20p data with 6,404 haplotypes. Bars
1128:    100- and 1,000-haplotype simulations and 1KGP data
1134:1.5$\times$ speedup for the 1000 Genomes Project with tsinfer 0.5 and 2.2$\times$ and 6.4$\times$
1135:improvements for the $n=10,000$ and $n=100,000$ simulations, respectively
1136:(Figure~\ref{fig:ch3-perf-elapsed}). The largest inference took about 69 hours with tsinfer 0.4 and
1137:only about 11 hours with tsinfer 0.5. The dramatic improvement in sample
1143:  \includegraphics[width=\textwidth]{../figures/ch3/inference_performance_elapsed.png}
1145:    and 0.5 in the simulated 28.1 Mb benchmark regions and in 1KGP chromosome
1165:  \includegraphics[width=\textwidth]{../figures/ch3/inference_performance_parallelism.png}
1193:with the pooled mismatch fraction across ancestors worsening by at most 0.073\%
1196:0.10\% relative to tsinfer 0.4 (42 additional focal sites at $\lambda = 2$).
1202:speed-up of up to almost $10\times$ was achieved. The more modest gains in
1336:demographic model when available. All simulations were done with the first 20 Mb
1337:of the chromosome 20 contig with 6,000 diploid individuals. The following
1343:      \item Mutation rate: $2.35\times10^{-8}$.
1348:      \item Mutation rate: $2.36\times10^{-8}$.
1353:      \item Mutation rate: $2.36\times10^{-8}$.
1358:      \item Mutation rate: the chromosome default of $1.29\times10^{-8}$.
1382:number of derived alleles $(G_{\mathrm{true}}, G_{\mathrm{obs}} \in \{0,1,2\})$. I obtained empirical genotype error
1401:For each true genotype $i \in \{0,1,2\}$, the total empirical error
1485:first 28.1 Mb of chromosome 20 under \texttt{OutOfAfrica\_3G09} with 50, 500,
1486:5,000 and 50,000 individuals split as close to equally between populations as
1495:and 192 threads and a memory reservation of 950,000 MB, although 196 instead of
1518:with one derived or at most two ancestral alleles. This retained 504,149 sites.

```
