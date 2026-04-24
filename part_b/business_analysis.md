# Part B — Business Case Analysis
## Scenario: Promotion Effectiveness at a Fashion Retail Chain

A fashion retailer operates 50 stores across urban, semi-urban, and rural locations. Each month, the marketing team runs one of five promotions: Flat Discount, BOGO, Free Gift with Purchase, Category-Specific Offer, and Loyalty Points Bonus. The goal is to determine which promotion maximises items sold in each store each month.

---

## B1. Problem Formulation

### B1(a) — ML Problem Formulation

**Target Variable:**
`items_sold` — the number of items sold at a given store in a given month under a given promotion.

**Candidate Input Features:**

| Feature Category | Examples |
|---|---|
| Store attributes | store_id, store_size, location_type (urban/semi-urban/rural), monthly footfall, competition_density |
| Promotion attributes | promotion_type (5 categories), promotion duration, discount depth |
| Calendar features | month, is_weekend, is_festival, is_month_end, season |
| Customer demographics | average customer age, income band, loyalty membership rate per store |
| Historical performance | rolling average items_sold per store, past response to each promotion type |

**Type of ML Problem:**
This is a **supervised regression** problem. The target variable (`items_sold`) is continuous and numerical, so regression is the appropriate framing. Each training example represents one store-month-promotion combination with a known historical outcome.

However, the ultimate business goal — choosing the *best* promotion for each store each month — can be framed as a **ranking or recommendation problem** on top of the regression: train the model to predict `items_sold` for all five promotion types given store and calendar context, then recommend the promotion with the highest predicted value.

**Justification:** Regression is preferred over classification here because the raw number of items sold matters, not just which bucket a result falls into. A model that predicts "Flat Discount will sell 310 items vs BOGO's 275" gives the marketing team much richer information than simply classifying the winner.

---

### B1(b) — Why Items Sold is a Better Target than Sales Revenue

**The argument for items sold over revenue:**

Sales revenue is the product of `items_sold × price`. Price varies dramatically across product categories and promotions — a Flat Discount directly reduces price, inflating items_sold but compressing revenue per unit. A BOGO promotion might appear to double revenue on paper while delivering the same margin as a single full-price sale.

Using revenue as the target would cause the model to conflate two distinct effects:
1. The promotion's ability to drive customer traffic and purchase volume (what we actually want to measure)
2. Price-level differences across product categories and promotion mechanics (a confounding variable)

`items_sold` isolates the **volume effect** of a promotion — a clean, consistent signal not contaminated by pricing strategy. This makes it a more reliable and interpretable target for evaluating promotion effectiveness.

**Broader principle — Target Variable Selection:**
This illustrates the principle that the **target variable should directly measure the outcome you want to optimise, not a downstream proxy that conflates multiple factors.** Revenue conflates volume (what the promotion drives) with price (an independent business decision). Choosing a cleaner target reduces noise, improves model interpretability, and ensures the model learns the right causal mechanism. In real-world ML projects, the most common modelling mistakes often stem from choosing a convenient target (what is easy to measure) rather than the right one (what the business actually wants to optimise).

---

### B1(c) — Modelling Strategy: Beyond a Single Global Model

**The problem with a global model:**
A single model trained across all 50 stores assumes that the relationship between promotions and items sold is the same everywhere. In reality, a Flat Discount may dramatically boost sales in a price-sensitive rural store but have little effect in an affluent urban store where customers respond better to a Free Gift. One global model would average out these heterogeneous effects, producing recommendations that are mediocre for every store rather than optimal for any.

**Proposed alternative: Location-Type Stratified Modelling with Store-Level Features**

Rather than one global model or 50 individual store models (which would have too little data per store), use a **stratified approach**:

1. **Train three separate models** — one per location type (urban, semi-urban, rural). Stores within the same location type share similar customer demographics, competition levels, and promotion sensitivities, so each model learns context-appropriate patterns.

2. **Include store-level features** (store_id as a category, store_size, footfall, competition_density) within each model so individual store characteristics still influence predictions within the group.

3. **Optional enhancement — Mixed Effects or Store Embeddings:** Use a single model with store-level random effects (mixed-effects regression) or learned store embeddings in a gradient boosting model. This allows sharing statistical strength across stores while still personalising predictions to each store's historical behaviour.

This strategy balances the bias-variance trade-off: a fully global model has high bias (underfits store heterogeneity), while 50 individual models have high variance (too little data each). The grouped approach captures meaningful structural differences while maintaining sufficient training data per model.

---

## B2. Data and EDA Strategy

### B2(a) — Joining the Four Tables

**The four tables and their keys:**

| Table | Grain | Key Columns |
|---|---|---|
| Transactions | One row per transaction | store_id, transaction_date, items_sold |
| Store attributes | One row per store | store_id, store_size, location_type, footfall, competition_density |
| Promotion details | One row per promotion-period | store_id, promotion_type, start_date, end_date |
| Calendar | One row per date | date, is_weekend, is_festival |

**Join sequence:**
1. Join `transactions` ← `store attributes` on `store_id` (many-to-one: many transactions per store)
2. Join result ← `promotion details` on `store_id` + date falling within promotion's start/end window
3. Join result ← `calendar` on `transaction_date = date` (many-to-one: many transactions per calendar day)

**Grain of the final modelling dataset:**
**One row = one store × one month × one promotion type.**

Before modelling, aggregate daily transactions up to monthly store-level records:
- Sum `items_sold` across all days in the month for each store
- Attach the promotion type active that month (one promotion per store per month)
- Attach store attributes (static, join directly)
- Attach calendar aggregations: count of weekends and festival days within that month

This monthly grain aligns with the business decision cadence — the marketing team decides promotions monthly, so the model must also operate at that level.

---

### B2(b) — EDA Strategy

**Analysis 1 — Items Sold Distribution by Promotion Type (Box Plot)**
Plot a box plot of `items_sold` for each of the five promotion types. Look for: which promotions have the highest median sales, which have the widest variance (inconsistent performance), and whether any promotion systematically underperforms. *Influence on modelling:* High-variance promotions are harder to predict and may need richer features or separate sub-models. Promotions with nearly identical distributions may be safely merged as a feature category.

**Analysis 2 — Promotion Performance by Location Type (Grouped Bar Chart)**
Cross-tabulate average `items_sold` by promotion type × location type. Look for: interaction effects — does BOGO work better in urban stores while Flat Discount works better in rural ones? *Influence on modelling:* Strong interaction effects justify including a `promotion_type × location_type` interaction feature, or directly support the stratified modelling strategy proposed in B1(c).

**Analysis 3 — Time Series of Items Sold per Store (Line Plot)**
Plot monthly items_sold over time for a sample of stores. Look for: trend (is overall sales growing?), seasonality (are December and festival months consistently higher?), and structural breaks (sudden shifts indicating a store renovation, competitor entry, or data quality issue). *Influence on modelling:* Confirmed seasonality justifies month and festival features; a trend justifies including year; structural breaks may require excluding affected periods from training data.

**Analysis 4 — Correlation Heatmap of Numerical Features vs Target**
Compute pairwise correlations between `items_sold` and numerical features (competition_density, footfall, is_festival, etc.). Look for: which features have the strongest linear relationship with the target, and whether any features are highly correlated with each other (multicollinearity). *Influence on modelling:* High multicollinearity between features (e.g., footfall and store_size) may require dropping one or using dimensionality reduction. Near-zero correlation features may be dropped to reduce noise.

**Analysis 5 — Missing Value and Data Quality Audit**
Check all four raw tables for missing values, duplicate store-month-promotion rows, and implausible values (e.g., items_sold = 0 or negative). *Influence on modelling:* Determines imputation strategies and flags data quality issues that could corrupt model training if left unaddressed.

---

### B2(c) — Handling the 80% No-Promotion Imbalance

**How this affects the model:**
If 80% of transactions occurred without any promotion, a model trained naively on this data will be biased toward predicting sales volumes typical of the no-promotion baseline. It will have seen relatively few examples of each of the five promotion types and may systematically underestimate promotional uplift. The model might learn that promotions matter less than they do, simply because it has overwhelmingly seen non-promotional examples during training.

**Steps to address it:**

1. **Separate the modelling objective clearly:** If the goal is to recommend among the five promotion types, restrict the training data to rows *with* a promotion only. The no-promotion baseline can be modelled separately as a reference benchmark ("promotion X lifts sales by Y% above no-promotion baseline for this store type").

2. **Engineer a promotional uplift feature:** Rather than predicting raw `items_sold`, compute `uplift = items_sold_with_promotion / avg_items_sold_no_promotion_for_that_store_and_month`. Training the model to predict uplift normalises for store baseline differences and focuses the model entirely on the incremental effect of each promotion type.

3. **Stratified sampling during training:** If keeping all data in a single model, oversample promotion-active periods or undersample no-promotion periods to rebalance the training distribution, ensuring the model sees sufficient and proportionate examples of each promotion type.

4. **Evaluate on promotion-only test records:** The business cares about ranking promotions against each other — evaluation metrics should be computed on promotion-active test records only, not averaged across all records where the no-promotion majority would dominate and mask poor promotion-specific performance.

---

## B3. Model Evaluation and Deployment

### B3(a) — Train-Test Split Setup and Evaluation Metrics

**How to set up the train-test split:**

With 3 years of monthly data across 50 stores, we have approximately 1,800 store-month records (50 stores × 36 months). The correct approach is a **temporal split**:

- **Training set:** Months 1–30 (January 2022 – June 2024, ~83% of data)
- **Test set:** Months 31–36 (July – December 2024, most recent 6 months)

For more robust evaluation, use **time-series cross-validation (walk-forward validation):** Train on months 1–12, test on month 13; then train on months 1–24, test on month 25; and so on. This produces multiple test folds without leaking future data, giving a more reliable estimate of true out-of-sample performance.

**Why random split is inappropriate:**
A random split would scatter future months (e.g., December 2024) into the training set and past months (e.g., January 2022) into the test set. The model would effectively be evaluated on predicting the past using knowledge of the future — a fundamental form of data leakage. In deployment, the model will always predict the future from the past, so evaluation must mirror this exact structure.

**Evaluation Metrics:**

| Metric | Business Interpretation |
|---|---|
| **RMSE** | Average prediction error in items, penalising large errors heavily. A RMSE of 30 means predictions are typically off by ~30 items. |
| **MAE** | Average absolute error; easier to communicate to stakeholders. "On average, our predictions are off by X items per store per month." |
| **MAPE** | Percentage error; allows fair comparison across stores of very different sizes (a large store and a small store both assessed on their own scale). |
| **Promotion Ranking Accuracy** | Percentage of store-months where the model correctly identifies the best-performing promotion. This most directly measures business value — recommendation correctness matters more than absolute accuracy. |

Promotion Ranking Accuracy is the most business-relevant metric: a model with moderate RMSE is still highly valuable if it consistently identifies the right promotion to deploy, even if its point estimates are imperfect.

---

### B3(b) — Explaining Different Recommendations Using Feature Importance

**The scenario:** The model recommends Loyalty Points Bonus for Store 12 in December and Flat Discount for Store 12 in March.

**Step 1 — Identify what changed between the two predictions:**
Store 12's attributes (size, location type, competition density) are constant across both months — these inputs are identical. The only features that differ are time-varying: month (12 vs 3), is_festival, is_weekend count, and any rolling historical averages for that seasonal period. The model's differing recommendation is driven entirely by how these time-varying features interact with each promotion type.

**Step 2 — Use SHAP values for instance-level explanation:**
SHAP (SHapley Additive exPlanations) values decompose a single prediction into the contribution of each individual feature — exactly what we need to explain two specific predictions. For each of the five promotion types in December and March:
- Compute SHAP values for `is_festival`, `month`, `is_weekend`, `competition_density`, and historical features
- Compare which features push the predicted `items_sold` higher for Loyalty Points Bonus in December vs Flat Discount in March

**Step 3 — Communicate findings to the marketing team:**

Present a clear narrative backed by SHAP evidence:

*"In December, `is_festival = 1` and `month = 12` strongly favour Loyalty Points Bonus. Historical data shows that existing loyalty customers make larger repeat purchases during the festive season — the model has learned that rewarding repeat behaviour is most effective when purchase intent is already high."*

*"In March, there are no festivals and footfall is seasonally lower. The model has learned that Flat Discounts are most effective at driving incremental volume from price-sensitive occasional shoppers who need a tangible incentive to make a discretionary purchase — this effect is strongest when the baseline motivation to buy is low."*

This SHAP-grounded narrative translates model mathematics into intuitive business logic that a marketing team can validate against their domain knowledge and act upon with confidence.

---

### B3(c) — End-to-End Deployment Process

**1. Saving the Trained Model**

Serialise the complete scikit-learn pipeline (preprocessor + model) using `joblib`:

```python
import joblib
joblib.dump(pipeline, 'promotion_recommender_v1.joblib')
```

Store versioned model artefacts in a cloud storage bucket (e.g., AWS S3 or Google Cloud Storage) alongside a metadata file recording: training date, training data date range, feature list, hyperparameters, and test set performance metrics. Never overwrite previous model versions — always increment the version number (v1, v2, etc.) so rollback to a previous model is always possible.

**2. Preparing and Feeding New Monthly Data**

At the start of each month, an automated pipeline (e.g., an Apache Airflow DAG or a scheduled Cloud Function) executes the following steps:

- Pull the previous month's transaction data from the data warehouse
- Join with store attributes and calendar tables using the same logic as during training
- Aggregate to store-month grain and apply the same feature engineering steps (date features, uplift calculation, encoding)
- Load the saved model: `pipeline = joblib.load('promotion_recommender_v1.joblib')`
- Generate predictions for all five promotion types for each of the 50 stores
- Output a recommendation table: one row per store, showing predicted items_sold for each promotion type and the final recommended promotion

**3. Monitoring for Model Degradation**

Implement a monitoring dashboard tracking the following signals monthly:

| Signal | What to Monitor | Retraining Threshold |
|---|---|---|
| **Prediction accuracy** | Actual vs predicted items_sold after each month | RMSE rises >20% above baseline → trigger review |
| **Data drift** | Distribution of input features vs training distribution (KL divergence or PSI) | PSI > 0.2 for any key feature → flag for investigation |
| **Concept drift** | Whether the promotion-to-sales relationship is shifting (e.g., new competitor, economic change) | Model recommendations consistently conflict with domain expert expectations |
| **Business outcome tracking** | Whether following model recommendations actually improves items_sold vs the historical no-model baseline | Tracked in monthly business review with marketing team |

**Retraining schedule:** Retrain every 6 months on a rolling window of the most recent 24 months of data, or immediately if any monitoring threshold is breached. Always validate the new model against a held-out recent period before replacing the production version, and use A/B testing across a random subset of stores to confirm the new model outperforms the old one in live conditions before full rollout.
