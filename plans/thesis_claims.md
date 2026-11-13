# Chapter 3 generated claim review

Every value below is generated from declared inputs. Detailed unrounded values, units and calculations are emitted to `statistics/ch3_claims.json`; supporting numerical tables are in `ch3_checks.json`. Thesis prose uses mechanical macro references. Literature estimates, mathematical constants and chapter 2 findings retain their existing attribution.

The original prose used 572.2% for the MAF-matched comparison, 4.2×/9.7× for the largest stage speedups, 6% for the largest phase-bin increase, and 0.22% for the frequency-bin mismatch upper bound. The retained data give 572.3866%, 4.2625×/9.7696×, 6.5389%, and 0.22156%, respectively. Generated presentation uses 572.4%, 4.3×/9.8×, and outward upper bounds of 7% and 0.23%. The demographic upper bound remains 0.024%; severe genotype-bin increase remains bounded by 1.3%. These are numerical/rounding corrections, not changes to filtering, methods or prose.

| TeX macro | Generated text | Units | Claim / calculation |
| --- | --- | --- | --- |
| `ChThreeToyFocalCarriers` | four | samples | Derived-carrier count at the fixed illustrated focal site |
| `ChThreeToyOldRemovalSite` | 4 | one-based site | Algorithm replay 0.4.1: removal of sample b |
| `ChThreeToyNewRemovalSite` | 6 | one-based site | Algorithm replay 0.5.1: removal of sample c |
| `ChThreeFocalSites` | 358,285 | count | Error-free 0.5.1; sites separately, ancestor IDs unique within replicate |
| `ChThreeTrueAncestors` | 137,123 | count | Error-free 0.5.1; sites separately, ancestor IDs unique within replicate |
| `ChThreeInferredAncestors` | 228,445 | count | Error-free 0.5.1; sites separately, ancestor IDs unique within replicate |
| `ChThreeRecentErrorFreePercent` | 62 | percent of focal-site overlaps | 0.5.1, observed DAF < 0.001; zero-error count / all focal sites |
| `ChThreeOldestErrorFreePercent` | 22 | percent of focal-site overlaps | 0.5.1, observed DAF >= 0.5; zero-error count / all focal sites |
| `ChThreeFrequencyMismatchIncreasePercent` | 0.23 | relative percent | Maximum across configured frequency bins of 100*(new pooled mismatch / old - 1); pooled errors / compared inference sites |
| `ChThreeEmpiricalExtentMb` | 26.0 | Mb | Maximum minus minimum empirical inference-site position across both versions |
| `ChThreeDemographyOldMedianMinKb` | 13.8 | kb | Minimum across demographic models of pooled focal-side median, 0.4.1 |
| `ChThreeDemographyOldMedianMaxKb` | 32.4 | kb | Maximum across demographic models of pooled focal-side median, 0.4.1 |
| `ChThreeDemographyNewMedianMinKb` | 10.6 | kb | Minimum across demographic models of pooled focal-side median, 0.5.1 |
| `ChThreeDemographyNewMedianMaxKb` | 26.1 | kb | Maximum across demographic models of pooled focal-side median, 0.5.1 |
| `ChThreeDemographyApproxErrorPercent` | 50 | percent | Mean per-model/version fraction of focal overlaps with errors, rounded to nearest configured percentage step for 'roughly' |
| `ChThreeDemographyMismatchIncreasePercent` | 0.024 | relative percent | Maximum across models of 100*(new pooled mismatch / old - 1) |
| `ChThreeZigzagMinimumIncreasePercent` | 86 | relative percent lower bound | Minimum across other models and both versions of 100*(Zigzag pooled mismatch / other - 1), floored for 'at least' |
| `ChThreeDemographyNonnegativeBoundPercent` | 75 | percent of focal-site sides | Complement of the configured lower central quantile; verified that a strict majority of model/version cases meet this bound |
| `ChThreeGenotypeCommonErrorPercent` | 0.215 | percent of diploid genotypes | Sum changed / sum total, true MAF strictly above configured threshold, lambda=1 |
| `ChThreeGenotypePublishedRelativePercent` | 572.4 | relative percent | 100*(MAF-matched simulated diploid error rate / cited published rate - 1); published reference is configuration, not a derived result |
| `ChThreeGenotypeAllErrorPercent` | 0.078 | percent of diploid genotypes | Lambda=1; sum changed / sum total; no frequency filter |
| `ChThreeGenotypeOldMedianKb` | 17.13 | kb | Lambda=1, 0.4.1; median over both focal-site sides |
| `ChThreeGenotypeNewMedianKb` | 14.80 | kb | Lambda=1, 0.5.1; median over both focal-site sides |
| `ChThreeGenotypeModerateSeverePercent` | 3.3 | percent of focal-site sides | Lambda=1, 0.5.1; overshoot strictly below -1e+07 bp / all sides |
| `ChThreeGenotypeExtremeSeverePercent` | 11.9 | percent of focal-site sides | Lambda=10, 0.5.1; overshoot strictly below threshold / all sides |
| `ChThreeGenotypeExtremeNonpositivePercent` | 75 | percent of focal-site sides | Lambda=10, 0.5.1; overshoot <= 0 / all sides |
| `ChThreeGenotypeUndershootBinIncreasePercent` | 1.3 | relative percent of bin count | Maximum 100*(new-old)/old over bins with midpoint below the severe threshold and lambda=0,1,2,5,10; common physical edges over all selected sides; zero-old bins undefined |
| `ChThreeGenotypeNegativeMedianMultiplier` | 5 | genotype multiplier | Smallest tested lambda with a negative median in both versions |
| `ChThreePhaseModerateSER` | 7.65 \times 10^{-4} | switches / adjacent heterozygous-site pairs | Aggregate phase-only profile spacing 2e+06 bp; verified against the declared phase-run export |
| `ChThreePhaseExtremeSER` | 7.65 \times 10^{-3} | switches / adjacent heterozygous-site pairs | Aggregate phase-only profile spacing 200000 bp; verified against the declared phase-run export |
| `ChThreePhaseOldMedianKb` | 7.66 | kb | Spacing 2e+06 bp, 0.4.1; pooled focal-side median |
| `ChThreePhaseNewMedianKb` | 5.79 | kb | Spacing 2e+06 bp, 0.5.1; pooled focal-side median |
| `ChThreePhaseExtremeSeverePercent` | 1.63 | percent of focal-site sides | Spacing 200000 bp, 0.5.1; sides strictly below severe threshold / all sides |
| `ChThreePhaseBinIncreasePercent` | 7 | relative percent of bin count | Maximum across all common physical bins and phase-only SERs of 100*(new-old)/old; zero-old bins undefined |
| `ChThreeCombinedErrorFreeDropPoints` | 30 | percentage points | 0.5.1, correctly polarised focal sites; zero-error fraction none minus all |
| `ChThreeCombinedBaselineMismatchPercent` | 0.046 | percent of compared inference sites | 0.5.1, correctly polarised Baseline profile; sum errors / sum overlap inference sites |
| `ChThreeCombinedErrorMismatchPercent` | 0.140 | percent of compared inference sites | 0.5.1, correctly polarised Error profile; sum errors / sum overlap inference sites |
| `ChThreeCombinedMismatchFold` | three | fold | Combined / none pooled mismatch fraction in updated version, correctly polarised focal sites |
| `ChThreeCombinedMaximumOvershootMb` | 7.63 | Mb | Maximum signed focal-side overshoot, correctly polarised combined profile, updated version |
| `ChThreeMaximumRegimeMismatchIncreasePercent` | 0.073 | relative percent | Maximum over quoted pure/combined regimes (lambda <=10), correctly polarised sites; whole-overlap pooled mismatch new/old - 1 |
| `ChThreeMaximumSevereFocalIncreasePercent` | 0.10 | relative percent of focal-site count | Maximum across quoted regimes, correctly polarised sites; focal site severe if at least one side is strictly below threshold; 100*(new count / old - 1) |
| `ChThreeAdditionalSevereFocalSites` | 42 | focal sites | New minus old counts in the regime attaining the maximum relative severe-focal increase |
| `ChThreeWorstSevereGenotypeMultiplier` | 2 | genotype multiplier | Genotype multiplier of regime attaining maximum relative severe-focal count increase |
| `ChThreeLargestAncestorSpeedup` | 4.3 | fold | Haplotypes=100000, match_ancestors; mean of paired per-repetition old/new wall-time ratios |
| `ChThreeLargestSampleSpeedup` | 9.8 | fold | Haplotypes=100000, match_samples; mean of paired per-repetition old/new wall-time ratios |
| `ChThreeEmpiricalSampleSpeedup` | 4.7 | fold | Haplotypes=6404, match_samples; mean of paired per-repetition old/new wall-time ratios |
| `ChThreeTenThousandSampleSpeedup` | 3.3 | fold | Haplotypes=10000, match_samples; mean of paired per-repetition old/new wall-time ratios |
| `ChThreeEmpiricalTotalSpeedup` | 1.5 | fold | Haplotypes=6404; sum three stages within each run, then mean paired old/new wall-time ratios |
| `ChThreeTenThousandTotalSpeedup` | 2.2 | fold | Haplotypes=10000; sum three stages within each run, then mean paired old/new wall-time ratios |
| `ChThreeLargestTotalSpeedup` | 6.4 | fold | Haplotypes=100000; sum three stages within each run, then mean paired old/new wall-time ratios |
| `ChThreeLargestOldHours` | 69 | hours | Haplotypes=100000, version=0.4; mean run total wall seconds / 3600 |
| `ChThreeLargestNewHours` | 11 | hours | Haplotypes=100000, version=0.5; mean run total wall seconds / 3600 |
| `ChThreeApproxLargestSampleSpeedup` | 10 | fold | Largest simulated sample-matching speedup rounded for 'almost' in conclusion |
| `ChThreeEmpiricalRetainedSites` | 504,149 | inference sites | Empirical benchmark num_sites, verified identical across stages, repetitions and versions |
| `ChThreeSimulationMb` | 20 | Mb | Fixed experimental setting from thesis configuration |
| `ChThreeSimulationIndividuals` | 6,000 | diploid individuals | Fixed experimental setting from thesis configuration |
| `ChThreeSimulationHaplotypes` | 12,000 | haplotypes | Fixed experimental setting from thesis configuration |
| `ChThreeEmpiricalIndividuals` | 3,202 | diploid individuals | Fixed experimental setting from thesis configuration |
| `ChThreeEmpiricalHaplotypes` | 6,404 | haplotypes | Fixed experimental setting from thesis configuration |
| `ChThreeSevereThresholdMb` | 10 | Mb | Fixed experimental setting from thesis configuration |
