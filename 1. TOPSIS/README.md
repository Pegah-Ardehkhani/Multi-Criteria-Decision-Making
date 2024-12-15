# TOPSIS (Technique for Order Preference by Similarity to Ideal Solution) <a href="https://colab.research.google.com/github/Pegah-Ardehkhani/Multi-Criteria-Decision-Making/blob/main/1.%20TOPSIS/TOPSIS.ipynb" target="_parent\"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> [![nbviewer](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.org/github/Pegah-Ardehkhani/Multi-Criteria-Decision-Making/blob/main/1.%20TOPSIS/TOPSIS.ipynb)

The **TOPSIS** (Technique for Order Preference by Similarity to Ideal Solution) method is a multi-criteria decision-making (MCDM) technique that helps evaluate and rank alternatives based on their proximity to an ideal solution. It is widely used for decision problems where several criteria must be considered simultaneously, often under conflicting requirements.

### How TOPSIS Works:

TOPSIS follows a systematic process to rank alternatives based on their distance to both the ideal and negative-ideal solutions. The main steps involved are:

1. **Normalize the Decision Matrix**: The decision matrix is normalized using the Euclidean norm (for each criterion), so that each value is scaled to a comparable level.
   
2. **Apply Weights**: Weights are applied to each criterion to reflect their importance in the decision-making process.

3. **Determine Ideal and Negative-Ideal Solutions**: The ideal solution is the best possible value for each criterion (maximum for benefit criteria, minimum for cost criteria). The negative-ideal solution is the worst possible value for each criterion.

4. **Calculate Distances**: The Euclidean distance of each alternative to both the ideal and negative-ideal solutions is calculated.

5. **Calculate Relative Closeness**: The relative closeness of each alternative to the ideal solution is computed by comparing the distances to the ideal and negative-ideal solutions.

6. **Rank the Alternatives**: Alternatives are ranked based on their relative closeness, where the higher the closeness, the better the alternative.

## Example: Step-by-Step Calculation

Let's solve a small example step by step. Assume we have 3 alternatives $(A_1, A_2, A_3)$ and 3 criteria $(C_1, C_2, C_3)$. The decision matrix is as follows:

| Alternatives | $C_1$  | $C_2$  | $C_3$  |
|--------------|-----|-----|-----|
| $A_1$           | 8   | 7   | 2   |
| $A_2$           | 5   | 3   | 7   |
| $A_3$           | 7   | 5   | 6   |

### Step 1: Normalize the Decision Matrix

Normalize the matrix by dividing each element by the Euclidean norm of its column.

#### Norm of each column:
- Norm of $C_1$: $(\sqrt{8^2 + 5^2 + 7^2} = \sqrt{64 + 25 + 49} = \sqrt{138} \approx 11.75)$
- Norm of $C_2$: $(\sqrt{7^2 + 3^2 + 5^2} = \sqrt{49 + 9 + 25} = \sqrt{83} \approx 9.11)$
- Norm of $C_3$: $(\sqrt{2^2 + 7^2 + 6^2} = \sqrt{4 + 49 + 36} = \sqrt{89} \approx 9.43)$

Now, normalize each element by dividing by the corresponding norm:

| Alternatives | $C_1$     | $C_2$     | $C_3$     |
|--------------|--------|--------|--------|
| $A_1$           | 0.68   | 0.77   | 0.21   |
| $A_2$           | 0.43   | 0.33   | 0.74   |
| $A_3$           | 0.59   | 0.55   | 0.64   |

### Step 2: Apply Weights

Assume the following weights for each criterion:
- $C_1$: 0.5
- $C_2$: 0.3
- $C_3$: 0.2

Multiply each normalized value by its corresponding weight:

| Alternatives | $C_1$     | $C_2$     | $C_3$     |
|--------------|----------|----------|----------|
| $A_1$            | 0.34     | 0.23     | 0.04     |
| $A_2$            | 0.22     | 0.10     | 0.15     |
| $A_3$            | 0.30     | 0.17     | 0.13     |

### Step 3: Find the Ideal and Negative Ideal Solutions

- **Ideal Solution** (best values for each criterion):  
  - $C_1$: max(0.34, 0.22, 0.30) = 0.34
  - $C_2$: max(0.23, 0.10, 0.17) = 0.23
  - $C_3$: max(0.04, 0.15, 0.13) = 0.15

  Thus, the **ideal solution** is (0.34, 0.23, 0.15).

- **Negative Ideal Solution** (worst values for each criterion):  
  - $C_1$: min(0.34, 0.22, 0.30) = 0.22
  - $C_2$: min(0.23, 0.10, 0.17) = 0.10
  - $C_3$: min(0.04, 0.15, 0.13) = 0.04

  Thus, the **negative ideal solution** is (0.22, 0.10, 0.04).

### Step 4: Calculate the Distances

Now, we calculate the Euclidean distance for each alternative from the ideal and negative-ideal solutions.

- **Distance from Ideal**:  
  
  $d(A1, \text{Ideal}) = \sqrt{(0.34 - 0.34)^2 + (0.23 - 0.23)^2 + (0.04 - 0.15)^2} = \sqrt{0.0121} = 0.11$
  
  $d(A2, \text{Ideal}) = \sqrt{(0.22 - 0.34)^2 + (0.10 - 0.23)^2 + (0.15 - 0.04)^2} = \sqrt{0.0484 + 0.0169 + 0.0121} = \sqrt{0.0774} \approx 0.28$
  
  $d(A3, \text{Ideal}) = \sqrt{(0.30 - 0.34)^2 + (0.17 - 0.23)^2 + (0.13 - 0.15)^2} = \sqrt{0.0016 + 0.0036 + 0.0004} = \sqrt{0.0056} \approx 0.075$

- **Distance from Negative Ideal**:
  
  $d(A1, \text{Negative Ideal}) = \sqrt{(0.34 - 0.22)^2 + (0.23 - 0.10)^2 + (0.04 - 0.04)^2} = \sqrt{0.0144 + 0.0169} = \sqrt{0.0313} \approx 0.18$
  
  $d(A2, \text{Negative Ideal}) = \sqrt{(0.22 - 0.22)^2 + (0.10 - 0.10)^2 + (0.15 - 0.04)^2} = \sqrt{0.0121} = 0.11$
  
  $d(A3, \text{Negative Ideal}) = \sqrt{(0.30 - 0.22)^2 + (0.17 - 0.10)^2 + (0.13 - 0.04)^2} = \sqrt{0.0064 + 0.0049 + 0.0081} = \sqrt{0.0194} \approx 0.14$

### Step 5: Calculate Relative Closeness

The relative closeness for each alternative is given by:

$\text{Relative Closeness} = \frac{d(A, \text{Negative Ideal})}{d(A, \text{Ideal}) + d(A, \text{Negative Ideal})}$

- For A1:  
  
  $\text{Relative Closeness}_1 = \frac{0.18}{0.11 + 0.18} = \frac{0.18}{0.29} \approx 0.62$
  
- For A2:  
  $\text{Relative Closeness}_2 = \frac{0.11}{0.28 + 0.11} = \frac{0.11}{0.39} \approx 0.28$
  
- For A3:  
  $\text{Relative Closeness}_3 = \frac{0.14}{0.075 + 0.14} = \frac{0.14}{0.215} \approx 0.65$

### Step 6: Rank the Alternatives

Based on the relative closeness, the rankings are:

1. **$A3$** (0.65)
2. **$A1$** (0.62)
3. **$A2$** (0.28)

Thus, the best alternative is **A3**, followed by **A1**, and the worst alternative is **A2**.

## How to Use the Code

### Install Dependencies

Make sure you have the required libraries installed:
```bash
pip install numpy matplotlib


```

### Using the TOPSIS Function

The following code snippet demonstrates how to use the `topsis` function to solve an MCDM problem:

```python
from topsis import topsis

# Define your decision matrix and weights
decision_matrix = [
    [8, 7, 2],
    [5, 3, 7],
    [7, 5, 6]
]

weights = [0.5, 0.3, 0.2]  # Criteria weights

# Run TOPSIS
topsis(decision_matrix, weights, plot=True, verbose=True)
```

This will:
1. Normalize the decision matrix.
2. Apply the weights.
3. Compute the ideal and negative ideal solutions.
4. Calculate the relative closeness and rank the alternatives.
5. Plot the relative closeness and distances from the ideal and negative-ideal solutions.

## Conclusion

TOPSIS is a powerful method for ranking alternatives in multi-criteria decision-making problems. By following the steps outlined above, you can easily apply this method to real-world decision problems and evaluate alternatives based on their proximity to an ideal solution. The provided code helps automate this process and visualize the results.
