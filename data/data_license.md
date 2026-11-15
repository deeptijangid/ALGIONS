# Dataset Provenance, Schema, and Licensing Documentation

## 1. Overview and Source
This dataset was synthesized specifically for the **Algions AI Labs Private Limited** E-Commerce Product Recommendation Engine assessment. It simulates realistic e-commerce user-item behavioral interaction streams, product catalog metadata, and user preference cohorts.

- **Generation Engine:** `src/data_generator.py`
- **Random Seed:** `42` (Fixed for 100% determinism and exact reproducibility)
- **Timeframe:** 90-day simulation window (2026-08-01 through 2026-10-30)

## 2. Licensing and Compliance
- **License:** Open Source / Creative Commons CC-BY 4.0 & MIT License compatible.
- **Privacy & PII Assessment:** **Zero Personal Identifiable Information (PII)** is used or stored. All identifiers (`USER_xxxx`, `PROD_xxxx`, `SESS_xxxx`) are synthetic hash surrogates. Demographic attributes are aggregated demographic buckets ('18-24', 'budget', etc.).
- **Prohibited Data Check:** No proprietary, customer-sensitive, scraped, or copyright-infringing user data is present.

## 3. Schema Definitions

### A. Products Catalog (`data/raw/products.csv`)
| Column | Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `item_id` | String | Unique product identifier (Primary Key) | `PROD_0001` |
| `title` | String | Product name including brand and variant | `Sony WH-1000XM5 Wireless Headphones` |
| `category` | String | Top-level taxonomic category (6 categories) | `Electronics` |
| `sub_category` | String | Granular sub-category classification | `Audio & Headphones` |
| `brand` | String | Manufacturing brand name | `Sony` |
| `price` | Float | Unit price in USD ($8.00 - $1499.00) | `349.99` |
| `rating` | Float | Average customer rating (1.0 to 5.0) | `4.8` |
| `review_count` | Integer | Total verified purchase reviews | `245` |
| `tags` | String | Comma-separated keyword feature tags | `wireless, noise-cancelling, bluetooth` |
| `description` | String | Detailed textual product specification | High-quality headphones... |

### B. Users Cohort (`data/raw/users.csv`)
| Column | Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `user_id` | String | Unique synthetic user identifier | `USER_0042` |
| `age_group` | String | Age bracket category | `25-34` |
| `gender` | String | Demographic category | `F` |
| `primary_category`| String | Historical highest-affinity product domain | `Electronics` |
| `secondary_category`| String | Secondary interest domain | `Books & Media` |
| `price_tier` | String | Price sensitivity tier (budget/mid/premium) | `mid-range` |
| `activity_tier` | String | Cohort activity level (power/moderate/casual) | `moderate` |
| `signup_date` | String | Cohort enrollment date (YYYY-MM-DD) | `2025-06-14` |

### C. Interactions Stream (`data/raw/interactions.csv`)
| Column | Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `session_id` | String | Unique browsing session identifier | `SESS_000124` |
| `user_id` | String | User performing the interaction | `USER_0042` |
| `item_id` | String | Product interacted with | `PROD_0001` |
| `event_type` | String | Behavioral funnel step (`view`, `cart`, `purchase`) | `purchase` |
| `event_weight` | Float | Implicit feedback score ($1.0, 2.5, 5.0$) | `5.0` |
| `timestamp` | Datetime | Strict point-in-time timestamp (UTC) | `2026-08-15 14:22:10` |

## 4. Key Behavioral Characteristics
1. **Power-Law Distribution (Zipfian):** Reflects empirical e-commerce long-tail dynamics where top 20% products capture over 65% of engagement.
2. **Funnel Drop-Off:** Views ($70\%$) $\rightarrow$ Carts ($20\%$) $\rightarrow$ Purchases ($10\%$).
3. **High Matrix Sparsity:** $> 97\%$ matrix sparsity, mirroring real-world recommender challenges.

