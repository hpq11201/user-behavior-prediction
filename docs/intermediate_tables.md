\# Intermediate Tables



\## 1. Overview



Three basic intermediate aggregation tables were constructed from the

cleaned user behavior dataset.



These tables provide reusable precomputed statistics for downstream

exploratory analysis and feature engineering.



The source dataset contains 6,213,379 cleaned interaction records.



\## 2. User-Level Summary Table



Output:



`data/interim/user\_summary.parquet`



Number of rows:



\- 10,000 users



Main fields include:



\- `user\_id`

\- `total\_behaviors`

\- `unique\_items`

\- `unique\_categories`

\- `active\_hours`

\- `first\_behavior\_time`

\- `last\_behavior\_time`

\- `view\_count`

\- `favorite\_count`

\- `cart\_count`

\- `purchase\_count`

\- `view\_ratio`

\- `favorite\_ratio`

\- `cart\_ratio`

\- `purchase\_ratio`



The table summarizes each user's activity level, product coverage,

behavior composition, and purchase tendency.



\## 3. Item-Level Summary Table



Output:



`data/interim/item\_summary.parquet`



Number of rows:



\- 2,876,947 items



Main fields include:



\- `item\_id`

\- `category\_id`

\- `total\_behaviors`

\- `unique\_users`

\- `first\_behavior\_time`

\- `last\_behavior\_time`

\- `view\_count`

\- `favorite\_count`

\- `cart\_count`

\- `purchase\_count`

\- `purchase\_rate`



The table provides basic item popularity, user coverage, and conversion

information.



\## 4. Time-Level Summary Table



Output:



`data/interim/time\_summary.parquet`



Number of rows:



\- 744 hourly periods



The dataset covers 31 days, with 24 hourly periods per day:



31 x 24 = 744



Main fields include:



\- `date`

\- `hour`

\- `total\_behaviors`

\- `unique\_users`

\- `unique\_items`

\- `view\_count`

\- `favorite\_count`

\- `cart\_count`

\- `purchase\_count`



The table supports later analysis of hourly activity patterns and

high-conversion time periods.



\## 5. Validation



All three intermediate tables were validated after construction.



\### Missing Values



\- User table missing values: 0

\- Item table missing values: 0

\- Time table missing values: 0



\### Behavior Count Consistency



Total behavior counts:



\- User table: 6,213,379

\- Item table: 6,213,379

\- Time table: 6,213,379



All values exactly match the cleaned dataset.



\### Purchase Count Consistency



Total purchase counts:



\- User table: 106,678

\- Item table: 106,678

\- Time table: 106,678



The three aggregated tables therefore remain fully consistent with the

cleaned interaction dataset.



\## 6. Conclusion



The user, item, and time intermediate tables have been successfully

constructed and validated.



They are ready to support downstream SQL analysis, exploratory data

analysis, and feature engineering.

