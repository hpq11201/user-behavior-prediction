\# Feature Effectiveness Report



\## 1. Objective



This report evaluates the effectiveness of the engineered features used in the

purchase prediction task.



The analysis focuses on three questions:



1\. Which features are most strongly associated with future purchase?

2\. Which features contribute most to the final predictive model?

3\. Are there strong correlations or redundant relationships between features?



The analysis uses the validation modeling dataset and the final

HistGradientBoosting model.



\---



\## 2. Features evaluated



The following features are included:



\- `total\_behaviors`

\- `view\_count`

\- `favorite\_count`

\- `cart\_count`

\- `active\_days`

\- `days\_since\_last\_behavior`

\- `seen\_historically`

\- `from\_category\_recall`



The prediction target is:



\- `label\_purchase`



where:



\- `1` indicates that the user purchased the item during the future label window

\- `0` indicates that no purchase occurred during the label window



\---



\## 3. Correlation analysis



A sampled validation dataset was used to calculate the feature correlation

matrix.



The correlation heatmap is saved at:



`reports/feature\_effectiveness/feature\_correlation\_heatmap.png`



The underlying correlation matrix is saved at:



`reports/feature\_effectiveness/feature\_correlation\_matrix.csv`



\### 3.1 Correlation with future purchase



The observed correlations with `label\_purchase` are:



| Feature | Correlation with purchase |

| --- | ---: |

| `cart\_count` | 0.1196 |

| `total\_behaviors` | 0.0914 |

| `view\_count` | 0.0814 |

| `active\_days` | 0.0616 |

| `seen\_historically` | 0.0413 |

| `favorite\_count` | 0.0393 |

| `from\_category\_recall` | -0.0304 |

| `days\_since\_last\_behavior` | 0.0072 |



Among the individual features, `cart\_count` has the strongest simple linear

correlation with future purchase.



This is consistent with the interpretation that adding an item to cart

represents a stronger purchase-intention signal than browsing alone.



However, the individual correlations are relatively modest. This suggests that

future purchase is not explained by one feature alone and that interactions

between features are important.



\---



\## 4. Feature correlation and redundancy



The heatmap shows several strong relationships between predictor variables.



\### 4.1 `total\_behaviors` and `view\_count`



The correlation between:



\- `total\_behaviors`

\- `view\_count`



is approximately:



`0.99`



This is expected because browsing represents the majority of recorded user

behavior and `view\_count` forms a large component of `total\_behaviors`.



Therefore, these two variables contain substantial overlapping information.



\---



\### 4.2 `active\_days` and `seen\_historically`



The correlation between:



\- `active\_days`

\- `seen\_historically`



is approximately:



`0.89`



A user-item pair that has historical interaction will generally have at least

one active day, so the two variables are naturally related.



\---



\### 4.3 `seen\_historically` and `from\_category\_recall`



The correlation between:



\- `seen\_historically`

\- `from\_category\_recall`



is approximately:



`-0.94`



This strong negative relationship reflects the construction of the candidate

set.



Historical candidates are typically exact user-item pairs already seen during

the feature window, whereas category-recalled candidates are used to introduce

additional items based on category-level interest.



These two candidate sources therefore represent substantially different parts

of the candidate population.



\---



\## 5. Positive vs negative sample comparison



The mean feature values were compared between:



\- negative samples (`label\_purchase = 0`)

\- positive samples (`label\_purchase = 1`)



The results are saved in:



`reports/feature\_effectiveness/feature\_class\_statistics.csv`



The main results are:



| Feature | Negative mean | Positive mean | Positive / Negative |

| --- | ---: | ---: | ---: |

| `total\_behaviors` | 0.8521 | 6.2487 | 7.33x |

| `view\_count` | 0.8035 | 5.1878 | 6.46x |

| `favorite\_count` | 0.0176 | 0.1929 | 10.93x |

| `cart\_count` | 0.0230 | 0.7107 | 30.86x |

| `active\_days` | 0.3704 | 1.5279 | 4.13x |

| `days\_since\_last\_behavior` | 2.8524 | 4.3756 | 1.53x |

| `seen\_historically` | 0.3338 | 0.9543 | 2.86x |

| `from\_category\_recall` | 0.6950 | 0.2487 | 0.36x |



\### 5.1 Strong behavioral intensity differences



Positive user-item pairs show much stronger historical interaction than

negative pairs.



For example:



\- total interaction count is approximately 7.3 times higher

\- view count is approximately 6.5 times higher

\- favorite count is approximately 10.9 times higher

\- cart count is approximately 30.9 times higher



This indicates that future purchase is strongly associated with prior

engagement intensity.



\---



\### 5.2 Cart behavior provides particularly strong separation



The largest positive-to-negative ratio is observed for:



`cart\_count`



Positive samples have an average cart count of approximately:



`0.71`



while negative samples have an average of only:



`0.02`



This supports the interpretation that cart behavior is a strong purchase

intention signal.



\---



\### 5.3 Category recall candidates are less likely to convert than historical pairs



The mean value of `from\_category\_recall` is:



\- approximately 0.69 for negative samples

\- approximately 0.25 for positive samples



This suggests that category-recalled candidates are useful for expanding

candidate coverage, but they generally have weaker purchase likelihood than

exact user-item pairs that were already observed historically.



This is consistent with the candidate-generation results, where category recall

improved coverage but also introduced many additional negative candidates.



\---



\## 6. Recency interpretation



The raw mean of `days\_since\_last\_behavior` is higher for positive samples than

for negative samples.



This result should not be interpreted directly as evidence that older

interactions increase purchase probability.



The feature uses:



`-1`



for category-recalled items that do not have direct historical user-item

interaction.



Because these recalled candidates are disproportionately negative, the

sentinel value reduces the negative-sample mean.



Therefore, the raw class mean of `days\_since\_last\_behavior` is affected by

candidate construction.



The feature should instead be interpreted jointly with:



\- `seen\_historically`

\- `from\_category\_recall`

\- interaction-count features



The model-based importance results provide a more reliable indication of the

feature's predictive contribution.



\---



\## 7. Permutation feature importance



Permutation importance was calculated using the final HistGradientBoosting

model.



The metric used was:



`average precision (PR-AUC)`



For each feature, the feature values were randomly shuffled and the resulting

decrease in PR-AUC was measured.



A larger decrease means that the model depends more strongly on that feature.



The feature importance figure is saved at:



`reports/feature\_effectiveness/permutation\_feature\_importance.png`



The detailed results are saved at:



`reports/feature\_effectiveness/permutation\_feature\_importance.csv`



The ranking is:



| Rank | Feature | Importance mean |

| ---: | --- | ---: |

| 1 | `total\_behaviors` | 0.6279 |

| 2 | `days\_since\_last\_behavior` | 0.1147 |

| 3 | `view\_count` | 0.1057 |

| 4 | `cart\_count` | 0.0461 |

| 5 | `from\_category\_recall` | 0.0047 |

| 6 | `favorite\_count` | 0.0029 |

| 7 | `active\_days` | 0.0023 |

| 8 | `seen\_historically` | 0.0000 |



\---



\## 8. Interpretation of feature importance



\### 8.1 `total\_behaviors`



`total\_behaviors` is by far the strongest model feature.



Its permutation importance is approximately:



`0.628`



This means that randomly disrupting this feature causes a large decline in

model ranking performance.



The result indicates that overall historical interaction strength is highly

informative for future purchase prediction.



\---



\### 8.2 `days\_since\_last\_behavior`



`days\_since\_last\_behavior` is the second most important feature.



Although its simple correlation with the target is weak, its permutation

importance is relatively high.



This indicates that recency becomes useful in combination with other features,

rather than through a simple linear relationship.



This is one reason the nonlinear HistGradientBoosting model outperforms the

Logistic Regression baseline.



\---



\### 8.3 `view\_count`



`view\_count` is also highly important.



However, it is strongly correlated with `total\_behaviors`.



Therefore, the two variables should not be interpreted independently.



The model may use the difference between overall interaction intensity and pure

browsing intensity to distinguish different behavior patterns.



\---



\### 8.4 `cart\_count`



`cart\_count` has lower permutation importance than the three features above,

but it shows the strongest simple correlation with purchase and the largest

positive-vs-negative mean ratio.



This suggests that cart behavior is highly meaningful as a direct purchase

intention signal, even though the model also relies strongly on broader

interaction patterns.



\---



\### 8.5 Low permutation importance does not mean no business value



Features such as:



\- `seen\_historically`

\- `active\_days`

\- `favorite\_count`



have relatively low permutation importance.



This does not necessarily mean they contain no useful information.



Permutation importance can underestimate the contribution of highly correlated

features because another correlated variable may retain similar information

after one feature is shuffled.



For example, `seen\_historically` is strongly related to:



\- `active\_days`

\- `total\_behaviors`

\- `from\_category\_recall`



Therefore, its information may already be represented by other variables.



\---



\## 9. Model comparison evidence



Feature effectiveness should also be evaluated through model performance.



The Logistic Regression baseline achieved:



\- ROC-AUC: 0.9203

\- PR-AUC: 0.032219

\- PR-AUC lift over random: 37.77x



The HistGradientBoosting model achieved:



\- ROC-AUC: 0.9281

\- PR-AUC: 0.049415

\- PR-AUC lift over random: 57.93x



PR-AUC improved by approximately 53%.



This supports the conclusion that the engineered features contain nonlinear

interactions that are useful for purchase prediction.



\---



\## 10. Main conclusions



The feature-effectiveness analysis leads to five main conclusions.



First, historical interaction intensity is the strongest predictive signal.



Second, cart behavior provides strong direct separation between purchased and

non-purchased user-item pairs.



Third, interaction recency contributes important nonlinear information.



Fourth, several features are highly correlated, especially

`total\_behaviors` and `view\_count`, so feature importance should not be

interpreted independently.



Fifth, the improvement from Logistic Regression to HistGradientBoosting

indicates that combinations and nonlinear interactions between behavioral

features improve purchase prediction.



Overall, the current feature set is effective for ranking purchase candidates,

but future improvements could include richer user-level, item-level and

category-level features.

