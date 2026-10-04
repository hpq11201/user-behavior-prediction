\# Raw Data Inspection Report



\## 1. Dataset Overview



The raw user behavior dataset was inspected before any cleaning or

transformation.



\- Raw file: `data/raw/user\_behavior.csv`

\- File size: 469.46 MB

\- Total records: 12,256,906

\- Unique users: 10,000

\- Unique items: 2,876,947

\- Unique categories: 8,916



\## 2. Raw Schema



The source CSV contains the following columns:



\- `time`

\- `user\_id`

\- `item\_id`

\- `item\_category`

\- `behavior\_type`



To align the raw dataset with the project schema, the following column

mapping is used:



\- `time` -> `timestamp`

\- `item\_category` -> `category\_id`



The standardized schema is therefore:



\- `timestamp`

\- `user\_id`

\- `item\_id`

\- `category\_id`

\- `behavior\_type`



\## 3. Missing Values



No missing values were detected in any of the five standardized fields.



| Field | Missing Records |

|---|---:|

| timestamp | 0 |

| user\_id | 0 |

| item\_id | 0 |

| category\_id | 0 |

| behavior\_type | 0 |



\## 4. Behavior Distribution



The dataset contains only the four expected behavior types.



| Behavior Type | Meaning | Records |

|---|---|---:|

| 1 | View | 11,550,581 |

| 2 | Favorite | 242,556 |

| 3 | Add to cart | 343,564 |

| 4 | Purchase | 120,205 |



Invalid behavior records: 0.



\## 5. Timestamp Validation



All timestamps were successfully parsed using the expected hourly format.



\- Invalid timestamp records: 0

\- Earliest timestamp: 2025-11-18 00:00:00

\- Latest timestamp: 2025-12-18 23:00:00



\## 6. Duplicate Records



The chunk-based inspection detected 1,073,158 duplicate rows within

individual chunks.



This value is not treated as the final global duplicate count because

duplicates may also occur across different chunks. Exact global

deduplication will therefore be performed during the formal cleaning

stage.



\## 7. Initial Data Quality Assessment



The raw dataset has good structural quality:



\- no missing values were detected;

\- all behavior values fall within the valid range 1-4;

\- all timestamps can be parsed successfully;

\- the raw schema can be mapped consistently to the project schema.



The main data quality issue identified at this stage is the presence of

a substantial number of duplicate interaction records. Global duplicate

detection and removal will be handled in the next data-cleaning step.

