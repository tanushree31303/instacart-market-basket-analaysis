# 🛒 Instacart Market Basket Analysis & Product Recommendation System

A complete, end-to-end Data Science and Machine Learning project that discovers grocery co-purchasing patterns and powers real-time product recommendations using the **Instacart Online Grocery Shopping Dataset**.



## 💡 What is this Project About?

When you shop on grocery websites like Instacart, Amazon Fresh, or Walmart, you often see:  
> *"Customers who bought this also bought..."* or *"Frequently bought together"*.

In a physical grocery store, items like salad dressings are placed next to lettuce so shoppers buy both. On an online grocery platform with over **49,000 products**, the system needs an intelligent way to suggest complementary items in real-time.

This project:
1. Analyzes **30,000 real grocery orders** to uncover buying affinities (e.g., limes & cilantro, sparkling waters, garlic & onions).
2. Implements and compares two pattern mining algorithms: **Apriori** and **FP-Growth**.
3. Builds an interactive **Grocery Product Recommendation Engine** that takes any active shopping cart and outputs the top 5 complementary recommendations.
4. Evaluates the recommender on a **held-out test set of 6,000 orders** using Information Retrieval metrics (**Precision@5**, **Recall@5**, **F1@5**, **Hit Rate@5**, and **Catalog Coverage**).

---

## 🧺 What is Market Basket Analysis (MBA)?

**Market Basket Analysis** is a data mining technique based on transaction history:
- If a customer adds item **A** to their cart, how likely are they to also add item **B**?
- **Antecedent ($A$)**: The item(s) already in the shopping basket.
- **Consequent ($B$)**: The item recommended to accompany $A$.
- **Rule**: $A \rightarrow B$ (*"If someone buys $A$, they are likely to buy $B$"*).

---

## 📊 Understanding the Metrics (In Simple Terms)

### 1. Association Rule Metrics

| Metric | Simple Definition | Formula | How to Interpret |
| :--- | :--- | :--- | :--- |
| **Support** | How often items appear together in all shopping baskets. | $\frac{\text{Count}(A \text{ and } B)}{\text{Total Baskets}}$ | `0.01` means 1% of all grocery orders contained both items. |
| **Confidence** | Given that item $A$ is in the cart, what is the chance that $B$ is also added? | $\frac{\text{Support}(A \cup B)}{\text{Support}(A)}$ | `0.25` means 25% of shoppers who bought $A$ also bought $B$. |
| **Lift** | How much more likely $B$ is bought when $A$ is in the basket, compared to pure chance. | $\frac{\text{Confidence}(A \rightarrow B)}{\text{Support}(B)}$ | **Lift = 1**: No relationship.<br>**Lift > 1**: Positive association ($A$ actively boosts $B$).<br>**Lift < 1**: Negative association (substitutes). |
| **Leverage** | The difference between observed co-occurrence and expected independence. | $\text{Support}(A \cup B) - (\text{Support}(A) \times \text{Support}(B))$ | Higher values indicate strong mutual dependency above random expectation. |
| **Conviction** | How strongly item $B$ depends on item $A$ being present. | $\frac{1 - \text{Support}(B)}{1 - \text{Confidence}(A \rightarrow B)}$ | High conviction means $B$ is rarely bought without $A$. |

---

### 2. Recommender System Evaluation Metrics

Unlike typical machine learning models where you predict True/False, a recommender suggests top $K$ items. We evaluate performance using **Leave-One-Out cross-validation** on real, held-out customer carts:

| Metric | Simple Meaning | Why It Matters |
| :--- | :--- | :--- |
| **Precision@5** | Out of the 5 items recommended, what fraction was actually bought by the customer? | Measures how relevant the recommendations were (low noise). |
| **Recall@5** | Out of all items the customer actually bought, what fraction did the system successfully guess in the top 5? | Measures how complete the recommendations were. |
| **F1@5** | The harmonic mean of Precision@5 and Recall@5. | Gives a single balanced accuracy score between Precision and Recall. |
| **Hit Rate@5** | Did the top 5 recommendations contain **at least one** item the customer purchased? (Yes = 1, No = 0). | Direct measurement of user satisfaction — did the user see something they wanted? |
| **Catalog Coverage** | What percentage of all available grocery products is the recommender able to suggest? | Prevents "echo chamber" recommendations where the system only recommends bananas to everyone. |

---

## 🗂️ Dataset & Data Pipeline

The project uses the official **Instacart Market Basket Analysis dataset**, which can be downloaded directly from Kaggle:
👉 **[Download Dataset from Kaggle (psparks/instacart-market-basket-analysis)](https://www.kaggle.com/datasets/psparks/instacart-market-basket-analysis)**

Download and extract the CSV files directly into the root project directory:

| File | Records / Shape | Description |
| :--- | :--- | :--- |
| `orders.csv` | 3,421,083 rows | Metadata for every order (user ID, order number, day of week, hour of day). |
| `order_products__train.csv` | 1,384,617 rows | Line items for evaluation orders, including `add_to_cart_order` and `reordered`. |
| `order_products__prior.csv` | 32,434,489 rows | Historical order items per user. |
| `products.csv` | 49,688 rows | Product catalog (`product_id`, `product_name`, `aisle_id`, `department_id`). |
| `aisles.csv` | 134 rows | Aisle classification. |
| `departments.csv` | 21 rows | Department taxonomy (produce, dairy, snacks, beverages, etc.). |

### Data Preparation Steps:
1. **Long-Tail Pruning**: Filtered to the **Top 250 most frequent products** to remove rare, unrepresentative items and optimize computation.
2. **Representative Sampling**: Sampled **30,000 unique orders** containing the top products.
3. **Train-Test Split (Preventing Data Leakage)**:
   - **Training Set (80%)**: 24,000 orders used exclusively to construct the transaction matrix and mine association rules.
   - **Held-Out Test Set (20%)**: 6,000 orders reserved untouched to test recommender accuracy.
4. **Boolean Transaction Matrix**: 24,000 orders $\times$ 250 products encoded as memory-efficient `bool` format (6.10 MB RAM).

---

## ⚙️ Algorithms Compared: Apriori vs. FP-Growth

Both algorithms were run on identical data with the exact same threshold: `min_support = 0.005` (items appearing in at least 0.5% of baskets = 120+ orders).

* **Apriori Algorithm (1994)**:
  - Uses the downward closure property: *any subset of a frequent itemset must also be frequent*.
  - Level-by-level candidate generation ($C_1 \rightarrow L_1 \rightarrow C_2 \rightarrow L_2 \dots$).
  - **Limitation**: Requires repeated database scans and creates millions of temporary candidate itemsets in memory.

* **FP-Growth Algorithm (2000)**:
  - Uses a compact **Frequent Pattern Tree (FP-Tree)** to store items in memory.
  - Requires only **two database passes**: Pass 1 counts frequencies; Pass 2 builds the tree.
  - **Advantage**: **Zero candidate generation**, runs dramatically faster, and uses far less RAM.

---

## 📈 Complete Project Metrics & Benchmark Results

### 1. Algorithm Performance Comparison
*(Source: `model_comparison.csv`)*

| Metric | Apriori | FP-Growth | Winner / Observation |
| :--- | :--- | :--- | :--- |
| **Execution Time** | **21.31 s** | **8.72 s** | **FP-Growth is ~2.5x faster** |
| **Peak Memory Usage** | **2,241.71 MB** (~2.24 GB) | **20.01 MB** | **FP-Growth uses 112x LESS memory!** |
| **Frequent Itemsets Found** | **441** | **441** | **100% Identical** (Mathematical guarantee) |
| • *1-itemsets* | 250 | 250 | Single items with $\ge 0.5\%$ support |
| • *2-itemsets* | 188 | 188 | Product pairs |
| • *3-itemsets* | 3 | 3 | Product triplets |
| **Association Rules Generated** | **362** | **362** | **100% Identical** (`min_lift = 1.10`) |
| **Mean Support of Rules** | **0.0078** (0.78%) | **0.0078** (0.78%) | Average co-occurrence |
| **Mean Confidence of Rules** | **0.1458** (14.58%) | **0.1458** (14.58%) | Average conditional probability |
| **Mean Lift of Rules** | **2.06** | **2.06** | Average 2.06x lift over random chance |
| **Maximum Lift Observed** | **8.93** | **8.93** | Strongest product affinity |
| **Candidate Generation Required?** | Yes (Iterative BFS) | No (Prefix FP-Tree) | FP-Growth is architecturally superior |

---

### 2. Top Discovered Association Rules (High Lift)
*(Source: `rules_fpgrowth.csv`)*

| Antecedent (In Basket) | Consequent (Recommended) | Support | Confidence | Lift | Interpretation |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Lime Sparkling Water** | **Sparkling Water Grapefruit** | 0.54% | 28.04% | **8.93** | Shoppers who buy one flavor of sparkling water are nearly 9x more likely to buy another flavor. |
| **Organic Yellow Onion** | **Organic Garlic** | 0.77% | 19.53% | **5.11** | Cooking base aromatics are bought together 5.1x more than random chance. |
| **Bunched Cilantro** | **Limes** | 0.52% | 28.22% | **4.98** | Classic Mexican cooking / salsa ingredients complement each other strongly. |
| **Organic Cilantro** | **Limes** | 0.97% | 27.54% | **4.86** | High confidence (27.5%) and strong lift (4.86x). |
| **Organic Cilantro** | **Organic Red Onion** | 0.58% | 16.43% | **4.42** | Fresh produce co-purchase. |
| **Large Lemon** | **Limes** | 1.49% | 19.47% | **3.43** | Citrus fruit affinity. |
| **Organic Strawberries, Bag of Organic Bananas** | **Organic Raspberries** | 0.58% | 18.78% | **3.58** | 3-item affinity: Organic smoothie / berry bundle. |

---

### 3. Recommendation Engine Evaluation (Held-Out Test Data)
*(Source: `recommendation_evaluation.csv`)*

Evaluated across **500 representative held-out test orders** (orders with $\ge 2$ items) using leave-one-out testing ($K=5$):

| Evaluation Metric | FP-Growth Recommender | Apriori Recommender | Popularity Baseline | Key Observation |
| :--- | :---: | :---: | :---: | :--- |
| **Precision@5** | **0.0220** (2.20%) | **0.0220** (2.20%) | **0.0372** (3.72%) | Popularity has slightly higher precision because bananas are in 17% of all baskets. |
| **Recall@5** | **0.1100** (11.00%) | **0.1100** (11.00%) | **0.1860** (18.60%) | Recommending global bestsellers captures top products easily. |
| **F1@5** | **0.0367** | **0.0367** | **0.0620** | Harmonic mean of Precision and Recall. |
| **Hit Rate@5** | **0.1100** (11.00%) | **0.1100** (11.00%) | **0.1860** (18.60%) | 11% of testing baskets got an exact match from association rules alone. |
| **Catalog Coverage** | **0.1560** (15.60%) | **0.1560** (15.60%) | **0.0360** (3.60%) | **Association rules achieve 4.3x higher catalog coverage** (recommends diverse, personalized items vs. only top 9 items). |

#### 🔍 Why Popularity Alone is NOT the Best Solution:
- A popularity recommender simply tells every customer to buy bananas, strawberries, and spinach, regardless of what is in their basket.
- While this achieves high hit rates (bananas appear in ~17% of all orders), it provides **zero personalized cross-selling value**.
- **The Ideal Production Architecture**: A **Hybrid Recommender** that fires high-lift association rules when strong context exists, and falls back to popularity only when no association rules fire.

---

## 🤖 How the Recommendation Engine Works

```
           [ Customer Active Shopping Basket ]
                          │
                          ▼
            [ Clean & Validate Product Names ]
                          │
                          ▼
            [ Query Association Rules Database ]
                          │
         ┌────────────────┴────────────────┐
         ▼                                 ▼
[ Rules Found (Lift > 1.10) ]     [ No Matching Rules ]
         │                                 │
         ▼                                 ▼
[ Rank by Lift & Confidence ]     [ Fallback to Top Bestsellers ]
         │                                 │
         └────────────────┬────────────────┘
                          ▼
             [ Remove Items Already in Cart ]
                          │
                          ▼
        [ Display Top-N Recommended Products ]
```

### Interactive Test Scenarios Executed in the Project:
1. **Scenario 1 (Mexican Cooking Basket)**: `['Limes', 'Organic Cilantro']`  
   $\rightarrow$ Recommended: *Organic Garlic, Organic Yellow Onion, Large Lemon, Bunched Cilantro*.
2. **Scenario 2 (Berry Smoothie Basket)**: `['Organic Strawberries', 'Bag of Organic Bananas']`  
   $\rightarrow$ Recommended: *Organic Raspberries, Organic Blueberries, Organic Baby Spinach*.
3. **Scenario 3 (Cold Beverage Basket)**: `['Lime Sparkling Water']`  
   $\rightarrow$ Recommended: *Sparkling Water Grapefruit* (Lift: **8.93**).
4. **Scenario 4 (Salad Basket)**: `['Organic Baby Spinach', 'Organic Avocado']`  
   $\rightarrow$ Recommended: *Organic Grape Tomatoes, Sweet Potato (Yam), Limes, Large Lemon*.
5. **Scenario 5 (Edge Case / Typo Handling)**: `['Organic Bananana', 'Apple iPhone 15', 'Organic Garlic']`  
   $\rightarrow$ Automatically flags unknown items and safely produces garlic-affinity recommendations.

---

## 📁 Project Structure & Saved Deliverables

```
mba/
│
├── Instacart_Market_Basket_Analysis.ipynb   # Master Jupyter Notebook with all 20 steps
│
├── frequent_itemsets.csv                   # 441 discovered frequent itemsets (length 1, 2, 3)
├── rules_fpgrowth.csv                      # 362 association rules mined via FP-Growth
├── rules_apriori.csv                       # 362 association rules mined via Apriori
├── model_comparison.csv                    # Benchmarks: execution time, memory, rule count
├── recommendation_evaluation.csv           # Recommender retrieval metrics (Precision, Recall, etc.)
│
├── orders.csv                              # Instacart orders dataset
├── order_products__train.csv               # Products in train orders
├── order_products__prior.csv               # Historical customer orders
├── products.csv                            # Product names, aisles, departments
├── aisles.csv                              # Aisle category mapping
├── departments.csv                         # Department category mapping
│
└── README.md                               # Project documentation & metrics guide
```

---

## 🚀 How to Run This Project

### 1. Prerequisites & Dataset Download
1. Install Python 3.9+ along with required libraries:
```bash
pip install pandas numpy matplotlib seaborn mlxtend jupyter
```
2. Download the dataset CSV files from [Kaggle: Instacart Market Basket Analysis](https://www.kaggle.com/datasets/psparks/instacart-market-basket-analysis) and place them in the project folder.

### 2. Launch the Jupyter Notebook
Navigate to the project directory and start Jupyter:
```bash
jupyter notebook Instacart_Market_Basket_Analysis.ipynb
```

### 3. Run the Recommender Interactively
Inside the notebook, Section 18 provides an interactive text box where you can input any grocery list (e.g., `Lime Sparkling Water` or `Organic Garlic, Large Lemon`) to get live recommendations!

---

## 💼 Key Business Insights & Recommendations

1. **FP-Growth for Production**:  
   FP-Growth produces identical results to Apriori while consuming **112x less memory** and running **2.5x faster**. For production deployment handling millions of orders, FP-Growth is the clear winner.
2. **High-Value Product Bundling**:  
   - Sparkling water flavors should be promoted together (Buy 1 Lime, get 50% off Grapefruit).
   - "Taco Night" bundles: Pair cilantro, limes, garlic, and onions on product detail pages.
3. **Dynamic Cart Recommendations**:  
   Place recommendation widgets at the cart review screen to drive basket expansion and boost **Average Order Value (AOV)** by 10%–20%.
4. **Smart Warehouse / Fulfillment Optimization**:  
   Store high-lift product pairs (e.g., cilantro & limes, garlic & onions) in adjacent warehouse shelves to reduce picker walking distance and order fulfillment time.
5. **Hybrid Architecture**:  
   Deploy a two-tiered system: Contextual association rules for specialized baskets, with popularity fallback for brand-new or single-item carts.

---

*Author: Data Science & AI Engineering Portfolio*  
*Dataset Source: [Kaggle - Instacart Market Basket Analysis (psparks)](https://www.kaggle.com/datasets/psparks/instacart-market-basket-analysis)*
