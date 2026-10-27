I recommend **paired non-inferiority testing**, with uncertainty estimated across independent simulation replicates. That addresses your scientific aim—showing that the new algorithm does not meaningfully worsen accuracy—more directly than obtaining a non-significant difference.

If you specifically want to claim that the versions are practically indistinguishable in either direction, use **equivalence testing** instead. A tiny difference can be statistically significant and still pass an equivalence test. Failing an ordinary significance test does not establish equivalence. [Lakens, 2017](https://doi.org/10.1177/1948550617697177)

There are two distinct issues in your data:

- **Practical importance:** with enough independent observations, a negligible difference can be detected precisely.
- **Dependence:** focal sites share genealogies, ancestors and overlapping sequence. Left/right observations also belong to the same comparison. Treating these as independent would exaggerate precision.

Crucially, I checked that the current dataset contains **one simulation replicate, `replicate=0`**. Its 14 error profiles are not independent genealogical replicates.

The population-genetics literature supports two approaches to uncertainty. For genomic observations, block jackknifing accounts for linkage—for example, Patterson et al. use it for statistics constructed as ratios of pooled sums. However, block size matters: a recent evaluation of qpAdm found that smaller blocks produced smaller standard-error estimates. [Patterson et al., 2012](https://pmc.ncbi.nlm.nih.gov/articles/PMC3522152/), [*Testing times*, 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11373510/)

For your simulation benchmark, **independent simulated datasets are the cleaner solution**. The original tsinfer accuracy evaluation averaged sequence-weighted statistics across 100 independent simulation replicates. General simulation-study guidance also recommends comparing methods on the same generated datasets and reporting Monte Carlo uncertainty. [Kelleher et al., 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6726478/), [Morris et al., 2019](https://doi.org/10.1002/sim.8086)

I would define your two outcomes as follows. Here, I interpret “genotype error rate” as errors in the inferred ancestral haplotypes; the input genotype errors are shared between versions.

| Outcome | Recommended primary measure |
|---|---|
| Ancestral-state errors | Your pooled burden: **total errors / total shared-overlap bp** |
| Negative overshoot | **Mean total inward boundary displacement per matched comparison**, including zeros |

For the second measure, define

\[
U_i=\max(0,-o_{\ell,i})+\max(0,-o_{r,i}).
\]

This measures how much true ancestral sequence is omitted at the two boundaries. Including zeros is important: comparing only observations with negative overshoot lets the membership of the analysed group change between versions. Positive overshoot should also remain separate so that improvements there cannot cancel increased undershoot.

As secondary outcomes, I would report the proportion with any undershoot and a severe-undershoot tail probability. Similar means can otherwise conceal different tails.

My proposed analysis is:

1. **Choose acceptable deterioration margins before formal testing.**  
   Express the error margin as additional errors per Mb of overlap, and the undershoot margin in bp per comparison—or use the pooled fraction of true ancestral span omitted. Base these margins on downstream consequences or an independently justified tolerance. There is no universal population-genetics threshold that makes a 1% or 5% difference negligible.

2. **Run both versions on identical independent simulations.**  
   Preserve the pairing of genealogies, mutations and injected errors. Use pilot replicate differences to determine how many additional replicates are needed for useful precision. The number of focal sites within one simulation is not the relevant replication count.

3. **Bootstrap whole paired simulation replicates.**  
   Resample the same replicate IDs for both versions. Recompute pooled error numerators and overlap denominators in every bootstrap sample; retain the ratio-of-sums definition. Calculate differences as **0.5 minus 0.4**, so positive values indicate deterioration.

4. **Compare confidence bounds with the margins.**  
   For non-inferiority at the 5% level, the **one-sided 95% upper confidence bound** must lie below the permitted increase. For two-sided equivalence, the **90% confidence interval** must lie entirely within the prespecified equivalence bounds. These are the confidence-interval forms of the corresponding tests. [Lakens, 2017](https://doi.org/10.1177/1948550617697177)

I would show paired differences and confidence intervals by frequency bin, with the acceptable-deterioration boundary drawn explicitly. Prespecify which bins support the main claim; simultaneous confidence bands are useful if you want a claim covering the entire frequency curve.

With the **current single 20 Mb simulation**, I would report observed effect sizes and treat block-jackknife intervals as exploratory. Ancestor comparisons can span multiple blocks, so assigning them to blocks by focal position does not automatically remove their dependence. Artificially downsampling until \(p>0.05\) would not solve either problem.

The eventual conclusion should be **“no practically important deterioration within the specified margins”**, provided the confidence bounds support it. That is a stronger and more relevant result than “the difference was not significant.”
