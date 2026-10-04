\# Data Processing Report



\## 1. Overview



This report documents the initial preprocessing and cleaning workflow

applied to the raw user behavior dataset.



The raw dataset contains 12,256,906 interaction records and covers

user browsing, favorite, add-to-cart, and purchase behaviors.



The cleaning workflow focuses on schema standardization, data validity

checks, duplicate removal, data type optimization, and Parquet

conversion.



\## 2. Raw Dataset



\- Raw file: `data/raw/user\_behavior.csv`

\- File size: 469.46 MB

\- Initial records: 12,256,906



The raw source schema is:



\- `time`

\- `user\_id`

\- `item\_id`

\- `item\_category`

\- `behavior\_type`



The source columns are standardized as follows:



\- `time` -> `timestamp`

\- `item\_category` -> `category\_id`



The final standardized schema is:



\- `timestamp`

\- `user\_id`

\- `item\_id`

\- `category\_id`

\- `behavior\_type`



\## 3. Data Quality Validation



Before duplicate removal, the dataset was checked for missing values,

invalid behavior types, and invalid timestamps.



\### Missing Values



No missing values were found in any required field.



| Field | Missing Records |

|---|---:|

| timestamp | 0 |

| user\_id | 0 |

| item\_id | 0 |

| category\_id | 0 |

| behavior\_type | 0 |



\### Behavior Validation



Valid behavior values are defined as:



\- 1 = View

\- 2 = Favorite

\- 3 = Add to cart

\- 4 = Purchase



No records with behavior values outside the valid range were found.



\- Invalid behavior records removed: 0



\### Timestamp Validation



The source timestamp format is `YYYY-MM-DD HH`.



All timestamps were successfully parsed.



\- Invalid timestamp records removed: 0

\- Earliest timestamp: 2025-11-18 00:00:00

\- Latest timestamp: 2025-12-18 23:00:00



\## 4. Duplicate Removal



Duplicate interactions were identified using the following four-field

business key:



\- `user\_id`

\- `item\_id`

\- `behavior\_type`

\- `timestamp`



A total of 6,043,527 duplicate interaction records were identified and

removed.



To verify that this rule did not remove valid records with different

category information, the raw dataset was grouped by the same

four-field business key and the number of unique category values was

checked.



Validation results:



\- Keys with multiple categories: 0

\- Maximum categories for one key: 1



This confirms that each four-field interaction key maps to only one

category, so the duplicate removal rule does not discard conflicting

category information.



\## 5. Cleaning Summary



| Metric | Records |

|---|---:|

| Initial records | 12,256,906 |

| Missing records removed | 0 |

| Invalid behavior records removed | 0 |

| Invalid timestamp records removed | 0 |

| Duplicate records removed | 6,043,527 |

| Final cleaned records | 6,213,379 |



\## 6. Data Type Optimization



To reduce memory usage, selected integer fields were converted to more

compact data types:



\- `user\_id`: int32

\- `item\_id`: int32

\- `category\_id`: int32

\- `behavior\_type`: int8

\- `timestamp`: datetime64



\## 7. Output Format



The cleaned dataset was saved in Parquet format:



`data/processed/user\_behavior\_clean.parquet`



Parquet was selected because it provides efficient storage and faster

analytical reads compared with the original CSV format.



\## 8. Post-Cleaning Validation



The cleaned Parquet dataset was reloaded and validated.



Validation results:



\- Final rows: 6,213,379

\- Missing values: 0

\- Exact duplicate rows: 0

\- Duplicate business keys: 0



The cleaned dataset therefore satisfies the current preprocessing

requirements and is ready for downstream intermediate table

construction and feature engineering.

