# Multi-Criteria Decision Making (MCDM)

<p align="center"> 
  <img width="600" height="400" src="https://i.ytimg.com/vi/7OoKJHvsUbo/maxresdefault.jpg"> 
</p>

This repository provides various algorithms and methods for Multi-Criteria Decision Making (MCDM), which is a set of approaches used to evaluate and prioritize alternatives based on multiple, often conflicting, criteria. These methods are commonly used in fields such as business, engineering, economics, healthcare, and environmental management, where decisions need to account for various factors.

## Introduction

Multi-Criteria Decision Making (MCDM) is essential when faced with decisions that involve multiple criteria, some of which may conflict with each other. This repository includes implementations of several popular MCDM techniques to help users systematically evaluate and rank alternative solutions based on multiple criteria.

The methods in this repository allow you to:

- Assign weights to criteria.
- Evaluate alternatives based on different factors.
- Rank alternatives to aid in decision-making.

## Features

- Implementations of various MCDM methods.
- Support for decision matrices.
- Flexible weight assignment for criteria.
- Integration with Python's numerical and data manipulation libraries.
- Easy-to-use interface for both simple and complex decision problems.

## Methods

This repository includes the following Multi-Criteria Decision Making methods:

1. **Weighted Sum Model (WSM)**
   - A linear aggregation method where alternatives are scored by summing the weighted values of each criterion.

2. **Analytic Hierarchy Process (AHP)**
   - A method that uses pairwise comparisons and a hierarchical structure to prioritize alternatives.

3. **Technique for Order Preference by Similarity to Ideal Solution (TOPSIS)**
   - A method that evaluates alternatives by calculating their distance from an ideal and a negative-ideal solution.

4. **Elimination and Choice Translating Reality (ELECTRE)**
   - A family of methods used to solve multi-criteria decision problems by eliminating inferior alternatives.

5. **PROMETHEE (Preference Ranking Organization Method for Enrichment Evaluations)**
   - A pairwise comparison method based on the preferences of decision-makers.

6. **Simple Additive Weighting (SAW)**
   - A method similar to WSM but with a focus on normalization of criteria values.

7. **VIKOR (Vlse Kriterijumska Optimizacija Kompromisno Resenje)**
   - A compromise ranking method that seeks to find the best compromise alternative.

8. **Fuzzy MCDM**
   - Adaptations of standard MCDM methods to handle uncertainty or imprecision in decision criteria.

## Installation

To install and use the package, follow these steps:

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/mcdm.git
    cd mcdm
    ```

2. Install dependencies:

    You can install the required Python dependencies via `pip`:

    ```bash
    pip install -r requirements.txt
    ```

    Alternatively, you can install the package directly:

    ```bash
    pip install .
    ```

## Usage

Once installed, you can use the MCDM methods directly by importing them into your Python scripts. Here's a quick example of using the **Weighted Sum Model (WSM)** method:

```python
from mcdm import WeightedSumModel

# Example decision matrix with alternatives and criteria
alternatives = ['A1', 'A2', 'A3']
criteria = ['C1', 'C2', 'C3']
decision_matrix = [
    [8, 7, 9],  # A1
    [6, 9, 8],  # A2
    [7, 8, 8]   # A3
]

# Weights for each criterion
weights = [0.5, 0.3, 0.2]

# Create an instance of the WSM model
wsm = WeightedSumModel()

# Rank alternatives
ranked_alternatives = wsm.rank_alternatives(decision_matrix, weights)
print(ranked_alternatives)
```

You can replace the `WeightedSumModel` with other methods like **AHP**, **TOPSIS**, **ELECTRE**, etc., depending on your requirements.

## Examples

Several examples are included in the `examples` folder. These examples demonstrate how to use the different methods with real-world decision matrices.

### Example 1: Using TOPSIS
```python
from mcdm import TOPSIS

# Decision matrix
decision_matrix = [
    [7, 8, 9],  # A1
    [6, 9, 7],  # A2
    [8, 7, 8]   # A3
]

# Criteria weights
weights = [0.4, 0.3, 0.3]

# Create TOPSIS instance and rank alternatives
topsis = TOPSIS()
ranked = topsis.rank_alternatives(decision_matrix, weights)
print(ranked)
```

### Example 2: Using AHP
```python
from mcdm import AHP

# Pairwise comparison matrix for criteria
criteria_matrix = [
    [1, 3, 0.5],
    [1/3, 1, 1/5],
    [2, 5, 1]
]

# Create AHP instance and calculate weights
ahp = AHP()
weights = ahp.calculate_weights(criteria_matrix)
print(weights)
```

## Contributing

We welcome contributions from the community! If you'd like to contribute to this repository, please follow these steps:

1. Fork the repository.
2. Create a new branch.
3. Make your changes.
4. Submit a pull request with a description of your changes.

Please ensure that your code follows the PEP 8 style guide and includes appropriate tests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The algorithms and methods in this repository are based on research in the field of Multi-Criteria Decision Making.
- We would like to acknowledge the authors and contributors to the original works of the MCDM methods included here.
