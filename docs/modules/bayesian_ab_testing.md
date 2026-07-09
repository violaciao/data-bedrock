# Module Guide: Bayesian A/B Testing (`05_measurement/`)

## Overview

Bayesian A/B testing answers experiment decision questions directly:
- How likely is treatment better than control?
- How large is the expected lift?
- How likely is the lift large enough to matter?
- What is the business risk of shipping now versus collecting more data?

The worked notebook focuses on subscription conversion and early revenue using
the synthetic `users.csv` and `orders.csv` data. It uses conjugate
Beta-Binomial models, so it stays lightweight and does not require PyMC.

## When to use Bayesian A/B testing

```
Experiment question?
├── Binary metric (converted / purchased / churned)
│   └── Beta-Binomial posterior
├── Revenue per user
│   ├── Purchase probability
│   │   └── Beta-Binomial posterior
│   └── Revenue among purchasers
│       └── Bayesian bootstrap
└── Launch decision
    └── Posterior probability + minimum practical lift + guardrails
```

Use this approach when stakeholders need a decision-ready statement such as:
"There is a 96% posterior probability treatment improves 30-day conversion, and
an 82% probability the relative lift is above 2%."

## Quick start

Open the notebook:

```bash
jupyter notebook 05_measurement/bayesian_ab_testing_subscription_illustration.ipynb
```

The notebook is self-contained. It defines reusable helpers inline:

```python
outcomes, revenue_col = make_user_level_outcomes(users, orders)
experiment = experiment.merge(outcomes, on="user_id", how="left")

samples = bayesian_beta_binomial_ab_test(
    experiment,
    outcome_col="converted_30d",
    alpha=1,
    beta=1,
    n_samples=100_000,
    seed=42,
)

summary = summarize_posterior_results(samples)
plot_conversion_posteriors(samples)
plot_lift_distribution(samples, lift_type="relative")
```

## Core model

For binary conversion, each user either converts or does not convert:

```text
theta_control   ~ Beta(alpha, beta)
theta_treatment ~ Beta(alpha, beta)

conversions_control   ~ Binomial(n_control, theta_control)
conversions_treatment ~ Binomial(n_treatment, theta_treatment)
```

Because the Beta prior is conjugate to the Binomial likelihood:

```text
theta | data ~ Beta(alpha + conversions, beta + non_conversions)
```

Posterior samples are used to estimate:
- **Absolute lift** = `theta_treatment - theta_control`
- **Relative lift** = `theta_treatment / theta_control - 1`
- **Win probability** = `P(theta_treatment > theta_control)`
- **Practical win probability** = `P(relative_lift > minimum_practical_lift)`
- **94% HDI** for conversion rates and lift

## Revenue model

Revenue is usually zero-inflated: many users pay nothing, while purchasers have
skewed order values. The notebook models revenue per user as:

```text
revenue_per_user = purchase_rate * revenue_given_purchase
```

Where:
- `purchase_rate` uses the same Beta-Binomial model as conversion.
- `revenue_given_purchase` uses a Bayesian bootstrap over positive revenue rows.

If no revenue column is present, the notebook prints a clear skip message and
continues with conversion analysis only. It detects common names including
`amount`, `amount_usd`, `revenue`, `total`, `price`, and `net_revenue`.

## Decision rule

The notebook uses a practical launch rule:

```text
Ship treatment if:
1. P(treatment > control) >= 0.95
2. P(relative_lift > minimum_practical_lift) >= 0.80
3. No guardrail metric is harmed
```

This is intentionally stricter than "probability treatment wins is above 50%".
A tiny lift can be real but not worth shipping. Always define the minimum
practical lift before reading the result.

## Prior sensitivity

The notebook compares:

| Prior | Interpretation |
|-------|----------------|
| `Beta(1, 1)` | Weak, flat prior; lets data dominate |
| `Beta(2, 8)` | Business-informed prior centered around 20% conversion |
| `Beta(10, 90)` | Stronger conservative prior centered around 10% conversion |

With large samples, these priors should converge toward similar results. With
small samples, prior choice matters more and should be documented.

## Causal caveat

The subscription walkthrough assigns `variant` after the fact:

```python
experiment["variant"] = np.where(
    rng.random(len(experiment)) < 0.5,
    "control",
    "treatment",
)
```

That makes the notebook an illustration of Bayesian A/B testing mechanics, not
evidence of a real causal product treatment. A real experiment must assign the
variant before users experience the onboarding or checkout flow, then measure
outcomes after assignment.

The notebook includes a clearly labeled synthetic-effect section that flips a
small number of treatment non-converters to converters. This is for learning
only, so readers can see how a winning posterior changes.

## Reporting

Always report:
1. Sample sizes and conversions in both variants
2. Observed conversion rates and observed lift
3. Prior choice and why it is reasonable
4. Posterior probability treatment wins
5. Probability lift exceeds the minimum practical lift
6. 94% HDI for absolute and relative lift
7. Guardrail metrics, especially churn or refund risk
8. Whether treatment assignment happened before outcomes were generated

## Notebook

| Notebook | What it shows |
|----------|--------------|
| `05_measurement/bayesian_ab_testing_subscription_illustration.ipynb` | End-to-end Bayesian subscription conversion and revenue test with caveats, posterior charts, decision table, synthetic treatment effect, and prior sensitivity |
