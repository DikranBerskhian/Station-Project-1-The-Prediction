# Station Project 1 — Waiter Tips Prediction

This project predicts historical restaurant tips in dollars using **LinearRegression** from scikit-learn. It completes the waiter-tips option of the supplied assignment.

## Start and run

Use Python 3.11 or newer. Extract this folder, open a terminal inside it, then run:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter notebook waiter_tips_prediction.ipynb
```

Choose **Kernel → Restart & Run All**. The notebook is already executed with outputs. The dataset and interactive charts are included; no dataset download or chart internet connection is needed.

To reproduce from the command line:

```bash
jupyter nbconvert --to notebook --execute --inplace waiter_tips_prediction.ipynb
```

## Files

- `waiter_tips_prediction.ipynb`: explained, executed workflow.
- `data/tips.csv`: real 244-row, seven-column dataset bundled with Plotly Express.
- `outputs/`: five standalone HTML charts, evaluation metrics, fold results, coefficients, test predictions, and saved example prediction.
- `requirements.txt`: exact versions used for the core runtime.
- `SUBMISSION.md`: GitHub PR steps and peer-review checklist.

## Data and features

Source: https://github.com/plotly/datasets/blob/master/tips.csv . The included copy is exported from `plotly.express.data.tips()`.

| Column | Meaning | Use |
|---|---|---|
| total_bill | Total bill in dollars including tax | Numeric input |
| tip | Recorded tip in dollars | Target |
| sex | Recorded payer gender label | Categorical input |
| smoker | Recorded smoking status | Categorical input |
| day | Day of week | Categorical input |
| time | Lunch/dinner | Categorical input |
| size | Number of people | Numeric input |

Gender and smoking labels are used for assignment completeness, not as causal explanations or endorsed deployment features. There are no missing values in the bundled data. One repeated row is removed conservatively before splitting; it may also represent a genuinely identical bill. There are 243 cleaned rows: 194 training and 49 test.

## Method and library choices

Pandas reads, checks, and summarizes the data; NumPy calculates RMSE and verifies finite predictions. Plotly Express `scatter()` explores bill/party-size relationships and prediction errors; `pie()` summarizes days. `train_test_split(test_size=0.2, random_state=42)` provides a reproducible holdout. `ColumnTransformer`, `SimpleImputer`, and `OneHotEncoder` preprocess numeric and categorical features inside a `Pipeline` to prevent preprocessing leakage. Linear regression is specified by the assignment, interpretable, and suitable as a first model for a continuous target. `KFold` and `cross_validate` provide training-only five-fold validation. `DummyRegressor` predicts the training mean for comparison.

## Actual executed results

| Metric | Result |
|---|---:|
| Training R² | 0.4452 |
| Held-out test R² (`model.score`) | 0.5186 |
| Test MAE | $0.7665 |
| Test RMSE | $1.0831 |
| Baseline test R² | -0.0107 |
| Baseline test MAE | $1.1483 |
| Mean five-fold training CV R² | 0.3616 |
| CV R² standard deviation | 0.0734 |
| Mean CV MAE | $0.7697 |

The assignment calls the score accuracy, but regression `model.score()` returns **R²**, not percent accuracy. MAE is the average absolute dollar error. The test set is evaluated after validation, with no test-based tuning. The model improves on the mean-tip baseline by MAE on this split; small-sample results remain uncertain.

Example: $30 bill, party of four, Female payer label, non-smoker, Sunday dinner → **$4.23 predicted tip**. The unrounded value and exact features are saved in `outputs/example_prediction.json`. No fixed example was provided in the assignment.

## Outputs at each step

The notebook displays the first five records, column types, missing counts, duplicate count, descriptive statistics, cleaned row count, two feature scatter plots, a day-composition pie chart, correlation, train/test sizes, each CV fold, metric JSON, prediction diagnostics, the example, and model coefficients. The saved CSV/JSON/HTML files preserve the numeric and visual results.

## Verification

Every notebook code cell was executed sequentially in a fresh Python process using IPython, with captured outputs. This environment blocks the socket connections required to launch a separate Jupyter kernel, so verification used in-process execution. Assertions check required columns, valid bill/tip values, positive party sizes, disjoint train/test indices, finite predictions, and correct prediction count. Re-running all cells reproduces the fixed-seed results. Outputs are regenerated in `outputs/`.

## Limits

This small historical sample is not representative of every restaurant. Service quality, location, and customer preferences are missing. Associations do not establish causation. A random holdout is not chronological forecasting. Linear regression may extrapolate poorly or produce negative tips; this is an educational prediction model, not a validated recommendation system. No serialized executable model is needed: rerun the notebook to train it.

See `SUBMISSION.md` for the remaining platform steps. A GitHub PR and real peer reviews cannot be fabricated; repository and peer PR URLs must be supplied.
