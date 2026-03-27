# Fairness Data Analysis & Binning Strategy

Based on the 2.84 million representative rows extracted from your Census dataset, here is the statistical breakdown and our recommended binning strategy for your fair clustering algorithms.

## 1. Age (`AGEP`)
The dataset contains ages ranging from **15 to 97**.

**Current Distribution:**
- 10th Percentile: 21
- **25th Percentile: 32**
- **50th Percentile (Median): 50**
- **75th Percentile: 65**
- 90th Percentile: 75

**Problem with current `[0, 18, 35, 55, 120]` bins:**
The `Youth (0-18)` group is virtually empty since the minimum age in this dataset is 15. Your `Senior (55+)` bin captures over 40% of the entire population. Heavily unbalanced bins make fair grouping much more distorted.

**Recommended Quartile Bins:**
To ensure natural balance across your groups, use quartiles:
* `Youth`: $\le 32$
* `YoungAdult`: $33 - 50$
* `Adult`: $51 - 65$
* `Senior`: $66+$

```python
# Updated Age Binning Code
df_core['AGE_BIN'] = pd.cut(df_core['AGEP'], 
                            bins=[0, 32, 50, 65, 120],
                            labels=['Youth', 'YoungAdult', 'Adult', 'Senior'])
```

---

## 2. Income (`PINCP`)
The dataset contains personal incomes ranging from **\$0 to \$1,686,200**.

**Current Distribution:**
- 10th Percentile: $0
- 25th Percentile: $9,600
- 33rd Percentile: $15,000
- **50th Percentile (Median): $30,000**
- 66th Percentile: $48,000
- **75th Percentile: $61,000**
- 90th Percentile: $107,800

**Problem with current limit `[35000, 75000]`:**
Over 55% of the entire population falls into your `Low` bucket (<$35k). Your `High` bucket (>$75k) contains less than 20% of the people. 

**Recommended Tercile Bins:**
If you want 3 buckets (Low, Med, High), using the 33% and 66% marks ensures equal representation:
* `Low`: $\le \$15,000$ (Captures part-time, unemployed, and low-wage)
* `Med`: $\$15,001 - \$48,000$ (Median wage earners)
* `High`: $>\$48,000$ (Upper tier)

```python
# Updated Income Binning Code
df_core['INC_BIN'] = pd.cut(df_core['PINCP'], 
                            bins=[-np.inf, 15000, 48000, np.inf], 
                            labels=['Low', 'Med', 'High'])
```

---

## 3. Sex (`SEX`)
The split is nearly perfectly balanced. 
- **Type 2 (Female, usually):** 1.45M (51.2%)
- **Type 1 (Male, usually):**  1.38M (48.8%)

**Recommendation:**
You can safely use `SEX` as an intersectional attribute without worrying about creating a severely small minority group that stalls your iterative rounding algorithm.

---

## 4. Race (`RAC1P`)
This attribute is heavily skewed, which is very typical for the US Census:
- **1 (White):** 1.93M (68.1%)
- **9 (Two+ races):** 281K (9.9%)
- **2 (Black):** 252K (8.9%)
- **6 (Asian):** 175K (6.1%)
- **8 (Other):** 155K (5.4%)
- **3, 4, 5, 7 (Native, Pacific Islander, etc.):** < 35K combined (< 1%)

**Recommendation:**
If you intersect `Race` with `Age` and `Income`, groups 3, 4, 5, and 7 will become so small that the LP relaxations in your algorithms might fail or suffer massive rounding violations.
**Action:** Collapse race into 3 macro categories before creating the `GROUP_ID`:
* `White (1)`
* `Black (2)`
* `Other (everything else)`

```python
# Example Race Collapse
df_core['RACE_MACRO'] = df_core['RAC1P'].apply(lambda x: 'White' if x == 1 else ('Black' if x == 2 else 'Other'))
```

---

## Final Recommendation for `GROUP_ID` (Intersectional Fairness)
When defining the subset to balance across your clusters, pick **two** attributes to cross. Crossing all 4 (Age $\times$ Income $\times$ Sex $\times$ Race) will result in $4 \times 3 \times 2 \times 3 = 72$ groups. Algorithms like Bera's and Böhm's struggle drastically when the number of groups $H$ scales up, drastically increasing the Price of Fairness.

Instead, start with **Age $\times$ Income** (12 groups) or **Sex $\times$ Race** (6 groups) to keep the dimensionality of the fairness constraint reasonable while still showcasing powerful intersectional fairness.
