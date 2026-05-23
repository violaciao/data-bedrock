# Decision: Bootstrap CI for non-parametric effect estimates

**Date:** January 2024
**Status:** Accepted

## Context

When reporting confidence intervals for the Mann-Whitney U test (or any non-parametric test), there is no closed-form solution for the CI on the median difference. We needed to choose an approach.

## Decision

Use **percentile bootstrap** with 5,000 resamples for the CI on median differences.

## Rationale

1. **No distributional assumptions** — Bootstrap works regardless of the underlying distribution, which matters for skewed metrics like revenue or session duration.

2. **Interpretability** — The percentile bootstrap CI is intuitive: it's the range of observed treatment effects across 5,000 random re-draws of the data.

3. **Correctness over speed** — 5,000 resamples takes ~100ms on typical experiment sizes. That's fast enough for interactive notebooks.

4. **Alternatives considered:**
   - *Hodges-Lehmann estimator* — Gives a CI for the "pseudo-median" of pairwise differences, not the sample median difference. Harder to explain to stakeholders.
   - *Asymptotic CI via CLT* — Would require the median to be asymptotically normal, which is violated for very skewed data.

## Tradeoffs

- Bootstrap CIs are slightly conservative (wider than necessary) for small samples (n < 30). In practice, A/B tests at this stage should have n > 1,000 per variant.
- The CI is not the same as a confidence interval for the Wilcoxon statistic. We document this distinction clearly in `stats.py`.

## When to use each CI method

| Metric | Recommended CI method |
|--------|----------------------|
| Conversion rate | Wilson score CI |
| Mean revenue | Welch t-test CI |
| Median revenue / skewed continuous | Bootstrap percentile CI |
| CUPED-adjusted metric | Welch t-test CI on adjusted values |
