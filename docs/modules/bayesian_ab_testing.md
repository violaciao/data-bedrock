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

The notebook also includes two workflow pieces that matter in real experiment
reviews:
- **Prior predictive checks** — inspect what the prior implies about relative
  lift before looking at outcomes.
- **Revenue decomposition** — split revenue per user into purchase probability
  and revenue among purchasers, so the team can see why revenue moved.

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

Before fitting the posterior, inspect the prior-predictive lift:

```python
prior_samples = beta_prior_predictive_lift(alpha=2, beta=8, n_samples=100_000)
plot_prior_predictive_lift([prior_samples])
```

This answers: "Before seeing the experiment, how large a lift did this prior
consider plausible?"

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

## Prior predictive checks

Prior sensitivity after seeing the data is useful, but it is not enough. A
prior should be inspected before outcome analysis because it encodes how
surprising a lift would be.

The notebook compares priors such as:

| Prior | What it says before seeing data |
|-------|---------------------------------|
| `Beta(1, 1)` | Almost any conversion rate is plausible; relative lift can be extremely wide |
| `Beta(2, 8)` | Conversion is probably low to moderate; large lifts are still possible |
| `Beta(10, 90)` | Conversion is expected near 10%; the prior is more conservative |

The key diagnostic is the **prior-predictive relative lift distribution**:

```text
theta_control_prior   ~ Beta(alpha, beta)
theta_treatment_prior ~ Beta(alpha, beta)
prior_relative_lift   = theta_treatment_prior / theta_control_prior - 1
```

Why this matters:
- A weak prior lets the observed data dominate, but can make very large lifts
  look plausible before the experiment.
- A stronger prior pulls lift toward zero, which is useful when most product
  changes are expected to have small effects.
- If the prior-predictive lift is too optimistic or too narrow, adjust the prior
  before reading the experiment result.

## Revenue model

Revenue is usually zero-inflated: many users pay nothing, while purchasers have
skewed order values. The notebook models revenue per user as:

```text
revenue_per_user = purchase_rate * revenue_given_purchase
```

Where:
- `purchase_rate` uses the same Beta-Binomial model as conversion.
- `revenue_given_purchase` uses a Bayesian bootstrap over positive revenue rows.

The notebook reports three revenue views:

| Component | Question answered |
|-----------|-------------------|
| Purchase-rate lift | Did treatment make more users buy? |
| Purchaser-value lift | Did treatment change spend among buyers? |
| Revenue-per-user lift | What is the combined monetization effect? |

This distinction matters. A treatment can increase purchase rate while lowering
average order value, or reduce purchase rate while increasing buyer quality. The
ship decision should use revenue per user for the overall business outcome, but
the decomposition explains the mechanism.

If no revenue column is present, the notebook prints a clear skip message and
continues with conversion analysis only. It detects common names including
`amount`, `amount_usd`, `revenue`, `total`, `price`, and `net_revenue`.

## Decision rule

The notebook uses a practical launch rule for the primary conversion metric:

```text
Ship treatment if:
1. P(treatment > control) >= 0.95
2. P(relative_lift > minimum_practical_lift) >= 0.80
3. No guardrail metric is harmed
```

This is intentionally stricter than "probability treatment wins is above 50%".
A tiny lift can be real but not worth shipping. Always define the minimum
practical lift before reading the result.

### How the decision is made

The decision table applies three gates:

| Gate | Why it exists | Example interpretation |
|------|---------------|------------------------|
| `P(treatment > control) >= 0.95` | Avoid shipping changes that are probably noise | "The posterior says treatment is very likely better." |
| `P(relative_lift > minimum_practical_lift) >= 0.80` | Avoid shipping tiny wins that do not matter commercially | "The lift is likely big enough to justify rollout work and risk." |
| Guardrail not harmed | Avoid trading conversion gains for worse churn, refunds, or retention | "The win does not come from lower-quality users." |

The output is not "significant / not significant." It is a product decision:

| Result pattern | Practical action |
|----------------|------------------|
| Passes all gates | Ship treatment, subject to operational readiness |
| High win probability, low practical-lift probability | Do not ship yet; the effect may be too small |
| Low win probability | Keep control or redesign the treatment |
| Guardrail harmed | Do not ship without deeper investigation |
| Wide HDI around lift | Continue collecting data if the decision is valuable enough |

### Why not decide from the 94% HDI alone?

An HDI is useful for showing the range of plausible lift values, but a business
decision usually needs threshold probabilities. For example, a 94% HDI may cross
zero while there is still a high probability of a small positive lift. That may
be useful for a cheap copy change and useless for a risky checkout rebuild.

The notebook therefore reports both:
- **HDI** — plausible range of effect sizes
- **Threshold probabilities** — probability the effect clears the business bar

## Revenue decision logic

For revenue, the primary decision metric is expected revenue per user:

```text
revenue_per_user = purchase_rate * revenue_given_purchase
```

Use the revenue posterior to answer:
- `P(treatment revenue per user > control revenue per user)`
- expected absolute revenue lift per user
- expected relative revenue lift
- 94% HDI for revenue lift

Do not ship from purchase-rate lift alone if revenue is the business goal. A
checkout change that creates more low-value purchases may improve conversion but
not improve revenue per user. Conversely, a pricing or packaging change may
reduce purchase rate but increase revenue per user enough to be worthwhile.

## Prior sensitivity

The notebook compares:

| Prior | Interpretation |
|-------|----------------|
| `Beta(1, 1)` | Weak, flat prior; lets data dominate |
| `Beta(2, 8)` | Business-informed prior centered around 20% conversion |
| `Beta(10, 90)` | Stronger conservative prior centered around 10% conversion |

With large samples, these priors should converge toward similar results. With
small samples, prior choice matters more and should be documented.

Use prior sensitivity as a robustness check:
- If all priors point to the same decision, the result is stable.
- If weak and strong priors disagree, the data is not decisive enough for a
  high-confidence launch.
- If the strong prior materially reduces the expected lift, report both the
  optimistic and conservative read.

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
