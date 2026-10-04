\# Literature Review



\## 1. Overview



This document summarizes several representative studies related to

e-commerce recommendation, user behavior modeling, and sequential

interest prediction.



The purpose of this review is not only to summarize the models, but also

to identify ideas that can be applied to the current user behavior

prediction project.



\---



\## 2. Wide \& Deep Learning for Recommender Systems



\### Paper



Cheng et al. (2016), \*Wide \& Deep Learning for Recommender Systems\*.



\### Main Problem



Traditional linear models are good at memorizing frequent feature

co-occurrences, while deep neural networks are better at generalizing to

previously unseen feature combinations.



The paper proposes combining both abilities in one model.



\### Core Idea



The framework contains two components:



\- Wide component:

&#x20; learns explicit feature interactions and memorizes frequent patterns.

\- Deep component:

&#x20; uses embeddings and neural networks to learn more general user-item

&#x20; relationships.



The two parts are trained jointly.



\### Relevance to This Project



This idea is useful because user purchase behavior may depend on both:



\- explicit historical patterns, such as repeated interaction with the

&#x20; same category;

\- more general latent relationships between users and products.



The current project can apply similar thinking when designing features.



For example:



\- user purchase ratio;

\- item popularity;

\- user-category interaction frequency;

\- user-item historical interaction indicators.



These explicit features can later be combined with learned embeddings in

deep learning models.



\### Key Takeaway



A strong recommendation or prediction system should balance

memorization of known interaction patterns with generalization to

unseen user-item combinations.



\---



\## 3. Deep Interest Network for Click-Through Rate Prediction



\### Paper



Zhou et al. (2017), \*Deep Interest Network for Click-Through Rate

Prediction\*.



\### Main Problem



Traditional deep CTR models often compress a user's historical behavior

into one fixed-length representation.



However, users usually have multiple interests, and different historical

behaviors are relevant to different target items.



\### Core Idea



DIN introduces a local activation mechanism.



Instead of giving all historical behaviors equal importance, the model

calculates how relevant each historical behavior is to the current

candidate item.



This creates a target-dependent user interest representation.



\### Relevance to This Project



This idea is highly relevant to the later modeling stage.



For example, a user may have interacted with products from many

categories.



When predicting whether the user will purchase a specific target item,

recent interactions with similar items or categories may be more useful

than unrelated historical behaviors.



Possible future features include:



\- number of interactions between a user and the target category;

\- recent interaction frequency with similar items;

\- time since the last interaction with the target category;

\- target-related behavior sequence features.



DIN is also directly relevant because it is one of the deep learning

models planned in the project outline.



\### Key Takeaway



User interest should not always be represented by one fixed vector.

Interest can depend on the specific target item being predicted.



\---



\## 4. Deep Interest Evolution Network for Click-Through Rate Prediction



\### Paper



Zhou et al. (2018), \*Deep Interest Evolution Network for Click-Through

Rate Prediction\*.



\### Main Problem



DIN captures target-specific user interests, but user interests also

change over time.



A user's current purchase intention may differ significantly from their

earlier interests.



\### Core Idea



DIEN models the evolution of user interests from historical behavior

sequences.



It contains mechanisms for:



\- extracting latent interests from user behaviors;

\- modeling how those interests evolve over time;

\- emphasizing interests related to the target item.



\### Relevance to This Project



The current dataset contains timestamps for every interaction, so user

behavior can naturally be organized as a time-ordered sequence.



This suggests that later feature engineering should consider temporal

information rather than only total counts.



Possible features include:



\- time since the user's last interaction;

\- number of interactions in the previous 1, 3, or 7 days;

\- recent purchase frequency;

\- recent category preference;

\- change in activity intensity over time.



\### Key Takeaway



User behavior is dynamic. Recent and evolving interests may provide more

predictive information than lifetime aggregate statistics alone.



\---



\## 5. Behavior Sequence Transformer for E-commerce Recommendation



\### Paper



Chen et al. (2019), \*Behavior Sequence Transformer for E-commerce

Recommendation in Alibaba\*.



\### Main Problem



Many recommendation systems treat user features independently and do not

fully use the sequential structure of historical behaviors.



However, the order and context of user interactions can contain useful

signals.



\### Core Idea



BST applies the Transformer architecture to user behavior sequences.



Self-attention allows the model to learn relationships among multiple

historical interactions without relying only on a fixed recurrent

structure.



\### Relevance to This Project



The dataset records user interactions over time and therefore supports

the construction of ordered behavior sequences.



Possible later sequence features include:



\- recent sequence of behavior types;

\- sequence of recently viewed items;

\- sequence of recently interacted categories;

\- transition patterns such as view -> cart -> purchase;

\- time gaps between consecutive behaviors.



Although the initial modeling stage may start with traditional machine

learning models, these sequence representations can later support LSTM,

GRU, DIN, or Transformer-based approaches.



\### Key Takeaway



The order of user actions contains information that cannot be fully

captured by simple aggregate features.



\---



\## 6. BERT4Rec: Sequential Recommendation with Bidirectional Transformers



\### Paper



Sun et al. (2019), \*BERT4Rec: Sequential Recommendation with

Bidirectional Encoder Representations from Transformer\*.



\### Main Problem



Traditional sequential recommendation methods often process user

behavior from left to right.



This may limit the quality of learned sequence representations.



\### Core Idea



BERT4Rec uses bidirectional self-attention to learn representations from

both directions of a behavior sequence.



It uses masked-item prediction during training.



\### Relevance to This Project



The model highlights the importance of learning contextual information

from complete user behavior sequences.



Although BERT4Rec is not currently included as a required model in this

project, it provides a useful extension direction for future work.



The concept also reinforces the importance of preparing clean,

chronologically ordered user behavior sequences during feature

engineering.



\### Key Takeaway



User behavior sequences can be treated similarly to language sequences,

where surrounding interactions provide context for predicting future

preferences.



\---



\## 7. Implications for the Current Project



The literature suggests several important directions for this project.



\### Aggregation Features



Traditional aggregate features remain useful, including:



\- behavior counts;

\- behavior ratios;

\- user activity levels;

\- item popularity;

\- conversion rates.



These features correspond to the current user and item intermediate

tables.



\### Target-Dependent Features



Inspired by DIN, later feature engineering should consider the

relationship between a user and the target item or category.



Examples include:



\- user-category interaction count;

\- target-category purchase ratio;

\- time since last interaction with the target category.



\### Temporal Features



Inspired by DIEN, recent behavior should be separated from lifetime

behavior.



Examples include:



\- 1-day, 3-day, and 7-day activity counts;

\- recency;

\- recent purchase frequency;

\- recent activity trend.



\### Sequential Features



Inspired by BST and BERT4Rec, the order of user behaviors can be useful.



Examples include:



\- recent behavior sequence;

\- behavior transition probabilities;

\- interaction time gaps;

\- repeated view-cart-purchase patterns.



\### Modeling Strategy



A reasonable modeling progression for this project is:



1\. traditional aggregate features;

2\. Logistic Regression;

3\. XGBoost / LightGBM;

4\. sequence construction;

5\. LSTM / GRU;

6\. DIN;

7\. potential Transformer-based extension.



This progression allows model complexity to increase gradually while

keeping the feature engineering process interpretable.



\---



\## 8. References



\- Cheng, H. T. et al. (2016). Wide \& Deep Learning for Recommender

&#x20; Systems.

\- Zhou, G. et al. (2017). Deep Interest Network for Click-Through Rate

&#x20; Prediction.

\- Zhou, G. et al. (2018). Deep Interest Evolution Network for

&#x20; Click-Through Rate Prediction.

\- Chen, Q. et al. (2019). Behavior Sequence Transformer for E-commerce

&#x20; Recommendation in Alibaba.

\- Sun, F. et al. (2019). BERT4Rec: Sequential Recommendation with

&#x20; Bidirectional Encoder Representations from Transformer.

