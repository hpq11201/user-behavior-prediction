\# Pandas vs Dask for Large-Scale Data Processing



\## 1. Overview



The project requires processing a relatively large user behavior dataset

containing more than 12 million raw interaction records.



Two candidate Python data-processing tools considered for the project

are Pandas and Dask.



This document compares their characteristics and explains why Pandas was

selected as the primary processing tool for the current dataset.



\---



\## 2. Pandas



Pandas is an in-memory data analysis library widely used for tabular

data processing.



Its main advantages include:



\- simple and mature API;

\- strong support for filtering, grouping, aggregation, and joins;

\- broad compatibility with machine learning libraries;

\- convenient debugging and interactive analysis;

\- efficient processing when the dataset fits into available memory.



The main limitation is that a normal Pandas DataFrame is primarily

processed in memory.



When the dataset becomes significantly larger than available memory,

operations may become slow or fail because of memory limitations.



\---



\## 3. Dask



Dask provides a DataFrame API similar to Pandas but supports partitioned

and parallel computation.



Instead of loading the entire dataset into memory at once, Dask can split

the data into multiple partitions and execute operations across several

CPU cores or machines.



Its main advantages include:



\- out-of-core processing;

\- parallel execution;

\- ability to process datasets larger than available memory;

\- relatively familiar syntax for Pandas users;

\- support for scaling from one machine to distributed environments.



However, Dask also introduces additional complexity.



Examples include:



\- lazy evaluation;

\- task scheduling overhead;

\- more difficult debugging;

\- some Pandas operations are not fully equivalent or equally efficient;

\- small and medium datasets may not benefit from distributed processing.



\---



\## 4. Comparison



| Dimension | Pandas | Dask |

|---|---|---|

| Execution model | Mainly in-memory | Partitioned and lazy |

| Parallel processing | Limited | Supported |

| Out-of-core processing | Limited | Supported |

| API simplicity | High | Medium |

| Debugging | Simple | More complex |

| Small/medium data | Very suitable | May add overhead |

| Very large data | Memory limited | More suitable |

| Distributed computing | Not native | Supported |

| ML ecosystem compatibility | Very strong | Good, sometimes requires conversion |



\---



\## 5. Dataset Characteristics in This Project



The raw dataset used in this project has the following characteristics:



\- CSV size: approximately 469.46 MB;

\- raw records: 12,256,906;

\- cleaned records: 6,213,379;

\- five main fields;

\- 10,000 unique users;

\- 2,876,947 unique items;

\- 8,916 unique categories.



Although the row count is relatively large, the dataset still fits into

the available memory of the current development machine.



The complete cleaning workflow, including global duplicate detection,

was successfully executed using Pandas without a memory error.



\---



\## 6. Practical Processing Strategy



Pandas was therefore selected as the main processing framework for the

current stage.



Different strategies were used depending on the task.



\### Raw Dataset Inspection



The raw CSV was inspected using chunked reading:



```python

pd.read\_csv(

&#x20;   file\_path,

&#x20;   chunksize=1\_000\_000,

)

