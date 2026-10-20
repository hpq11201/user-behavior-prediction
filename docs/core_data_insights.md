\# Core Data Insights



\## 1. Overview



This analysis summarizes the main behavioral and predictive insights obtained

from the user behavior dataset.



The analysis is based on:



\- 12,256,906 cleaned behavior records

\- 10,000 users

\- 2,876,947 items

\- four behavior types:

&#x20; - View

&#x20; - Favorite

&#x20; - Cart

&#x20; - Purchase

\- SQLite-based behavioral analysis

\- strict user-item funnel analysis

\- feature engineering

\- candidate generation

\- Logistic Regression baseline

\- HistGradientBoosting purchase prediction model



The main objective is to understand user behavior patterns, identify conversion

bottlenecks, and provide practical recommendations for product operation and

purchase prediction.



\---



\## 2. User behavior patterns



\### 2.1 Browsing dominates user activity



The overall behavior distribution is highly concentrated in browsing activity.



Total behavior counts:



\- View: 11,550,581

\- Favorite: 242,556

\- Cart: 343,564

\- Purchase: 120,205



Views account for the majority of all user interactions.



This indicates that most user-item interactions remain at the browsing stage,

while only a small proportion progress to stronger purchase-intention

behaviors such as cart addition and purchase.



\---



\### 2.2 User activity is relatively stable but contains a strong peak



Daily active users are generally relatively stable across most dates.



However, 2025-12-12 shows a clear behavioral peak.



On this date:



\- Views: 641,507

\- Favorites: 10,446

\- Carts: 24,508

\- Purchases: 15,251

\- DAU: 7,720



The observed increase is substantially higher than surrounding dates.



The available dataset does not include external campaign or promotion

information, so the cause of the peak cannot be confirmed from the current

data alone.



However, this date represents an important high-activity period and may reflect

a temporary change in user purchasing behavior.



\---



\### 2.3 Stronger historical interaction is associated with future purchase



The feature analysis shows clear differences between future-purchase and

non-purchase user-item pairs.



Average historical behavior:



| Feature | Negative samples | Positive samples |

| --- | ---: | ---: |

| Total behaviors | 0.85 | 6.25 |

| Views | 0.80 | 5.19 |

| Favorites | 0.02 | 0.19 |

| Carts | 0.02 | 0.71 |

| Active days | 0.37 | 1.53 |



Future-purchase pairs therefore show substantially stronger historical

engagement.



Among these features, cart behavior has particularly strong discrimination.



The average cart count for positive samples is approximately 30.9 times the

average for negative samples.



This suggests that cart behavior is a strong indicator of purchase intention.



\---



\### 2.4 Interaction intensity is the strongest predictive signal



Permutation feature importance from the HistGradientBoosting model shows the

following ranking:



| Feature | Permutation importance |

| --- | ---: |

| `total\_behaviors` | 0.6279 |

| `days\_since\_last\_behavior` | 0.1147 |

| `view\_count` | 0.1057 |

| `cart\_count` | 0.0461 |

| `from\_category\_recall` | 0.0047 |

| `favorite\_count` | 0.0029 |

| `active\_days` | 0.0023 |

| `seen\_historically` | 0.0000 |



The dominant feature is `total\_behaviors`, indicating that the overall strength

of historical user-item interaction is highly informative for purchase

prediction.



Recency and browsing intensity also provide important additional signals.



\---



\## 3. Conversion bottlenecks



\### 3.1 User-level funnel results can be misleading



When behavior stages are counted independently at the user level, the apparent

conversion rates are high because users only need to complete each behavior

somewhere in the observation period.



For example:



\- View users: 10,000

\- Cart users: 8,614

\- Purchase users: 8,886



This approach does not guarantee that the behaviors relate to the same item or

occur in chronological order.



Therefore, independent user-level counts should not be interpreted as a strict

purchase funnel.



\---



\### 3.2 Strict user-item funnel shows substantial conversion loss



A stricter funnel was constructed at the user-item level.



The behavior sequence requires:



1\. the user views an item

2\. the same user later carts the same item

3\. the same user later purchases the same item



The strict funnel results are:



\- View pairs: 4,677,501

\- View → Cart pairs: 273,899

\- View → Cart → Purchase pairs: 58,424



Corresponding conversion rates:



\- View → Cart: 5.86%

\- Cart → Purchase: 21.33%

\- View → Purchase through cart: 1.25%



The largest conversion loss occurs between viewing and cart addition.



This suggests that many users explore items without progressing to a strong

purchase-intention stage.



\---



\### 3.3 Cart-to-purchase conversion is much stronger than view-to-purchase



Daily behavior-count conversion analysis also shows that cart behavior is much

closer to purchase than browsing behavior.



Typical View → Purchase rates are approximately around 1% on many dates,

whereas Cart → Purchase rates are substantially higher.



On 2025-12-12:



\- View → Purchase: 2.38%

\- Cart → Purchase: 62.23%



This further supports the interpretation that cart behavior is a high-value

purchase-intention signal.



\---



\### 3.4 Candidate generation is currently a major system bottleneck



The original historical user-item candidate strategy covered only:



\- 8,109 of 35,288 future purchase pairs

\- coverage: 22.98%



Adding category-based popular-item recall improved coverage:



| Strategy | Coverage |

| --- | ---: |

| Historical only | 22.98% |

| Category Top-5 | 23.91% |

| Category Top-10 | 24.50% |

| Category Top-20 | 25.21% |



Top-10 was selected because it provided a better balance between candidate size

and coverage.



However, even the selected strategy covers only approximately one quarter of

all future purchase pairs.



This means that the ranking model cannot predict many purchases because those

items are never included in the candidate set.



Therefore, candidate generation is currently one of the largest end-to-end

performance bottlenecks.



\---



\## 4. Purchase prediction insights



\### 4.1 Logistic Regression provides a strong baseline



The Logistic Regression model achieved:



\- ROC-AUC: 0.9203

\- PR-AUC: 0.032219

\- random baseline PR-AUC: 0.000853

\- PR-AUC lift over random: 37.77x



Although the absolute PR-AUC is numerically small, the dataset is extremely

imbalanced, with only 0.0853% positive samples in the validation candidate set.



Therefore, the large lift over the random baseline indicates meaningful ranking

ability.



\---



\### 4.2 HistGradientBoosting improves nonlinear purchase prediction



HistGradientBoosting achieved:



\- ROC-AUC: 0.9281

\- PR-AUC: 0.049415

\- PR-AUC lift over random: 57.93x

\- Precision at threshold 0.5: 5.04%

\- Recall at threshold 0.5: 43.40%

\- F1 at threshold 0.5: 0.0903



Compared with Logistic Regression, PR-AUC increased by approximately 53%.



This suggests that nonlinear interactions between behavioral features contain

additional predictive information.



The HistGradientBoosting model was therefore selected as the stronger final

model.



\---



\### 4.3 High-scoring candidates have substantially higher purchase rates



Global Top-K evaluation shows strong ranking concentration.



For HistGradientBoosting:



| Top-K | Precision@K | Lift@K |

| ---: | ---: | ---: |

| 100 | 15.00% | 175.86x |

| 500 | 15.40% | 180.55x |

| 1,000 | 16.00% | 187.59x |

| 5,000 | 11.32% | 132.72x |

| 10,000 | 9.89% | 115.95x |



The overall validation purchase rate is only 0.0853%.



Therefore, the model successfully concentrates purchases within the

highest-ranked candidate groups.



\---



\### 4.4 Per-user recommendation performance is also improved



For HistGradientBoosting:



| Recommendation size | Precision@K | Recall@K | Hit Rate@K |

| ---: | ---: | ---: | ---: |

| Top-1 | 5.92% | 6.77% | 17.01% |

| Top-5 | 4.02% | 22.98% | 45.23% |

| Top-10 | 3.11% | 35.48% | 60.38% |



For users who made at least one purchase during the validation window, the

Top-10 recommendation list contains at least one purchased item for

approximately 60.4% of users.



This indicates that the model has practical value as a user-item ranking tool.



\---



\## 5. Product operation recommendations



\### 5.1 Prioritize users with recent and repeated interaction



Users with stronger historical interaction are substantially more likely to

purchase.



Operational strategies can therefore prioritize user-item pairs with:



\- repeated browsing

\- recent interactions

\- cart additions

\- activity across multiple days



These users may be more suitable for personalized reminders, recommendation

exposure, or targeted conversion campaigns.



\---



\### 5.2 Cart behavior should be treated as a high-intent signal



Cart behavior shows both strong direct correlation with purchase and a large

difference between positive and negative samples.



Therefore, cart users can be treated as a high-priority conversion group.



Potential actions include:



\- cart reminders

\- personalized item recommendations

\- price or stock notifications

\- limited-time incentives where appropriate



The dataset itself does not contain pricing or campaign information, so the

effectiveness of these strategies would need to be validated separately.



\---



\### 5.3 Improve the View → Cart stage



The strict funnel shows that only 5.86% of viewed user-item pairs progress to

cart addition.



This is the largest funnel loss.



Operational analysis should therefore focus on factors that may prevent users

from progressing beyond browsing, such as:



\- product relevance

\- product-page information

\- recommendation quality

\- pricing competitiveness

\- user-product matching



These causes cannot be directly verified from the current dataset because the

available data contains behavioral records but not detailed product-page,

pricing, or campaign information.



\---



\### 5.4 Improve candidate generation before heavily tuning the ranking model



The ranking model already performs well within the current candidate set.



However, Top-10 category recall covers only 24.50% of all future purchase

pairs.



Future system improvements should therefore prioritize candidate generation.



Potential future directions include:



\- recent globally popular items

\- user-category preference strength

\- item co-occurrence

\- users with similar interaction patterns

\- category-level purchase popularity

\- time-decayed popularity

\- collaborative-filtering candidates



Improving recall at the candidate stage may provide larger end-to-end gains than

small additional ranking-model improvements.



\---



\### 5.5 Treat high-activity dates separately in operational analysis



2025-12-12 shows a clear increase in views, carts and purchases.



Model performance and conversion behavior during such high-activity periods may

differ from normal days.



Future analysis can therefore compare:



\- normal-period behavior

\- high-activity-period behavior



This can help determine whether separate recommendation or conversion

strategies are needed under unusually high demand.



\---



\## 6. Final conclusion



The dataset shows a clear behavioral progression from low-intent browsing to

higher-intent cart and purchase activity.



Historical interaction intensity, interaction recency, browsing frequency and

cart behavior all contribute to future purchase prediction.



The strict funnel identifies View → Cart as the main conversion bottleneck.



HistGradientBoosting provides stronger ranking performance than Logistic

Regression and achieves a PR-AUC lift of 57.93x over the random baseline.



However, the current candidate-generation strategy covers only 24.50% of future

purchase pairs.



Therefore, the main next-stage opportunity is not only further model tuning,

but improving candidate recall and enriching user, item and category-level

features.

