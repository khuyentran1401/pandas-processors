# pandas-processors

A Python library that provides utilities to process your pandas DataFrame.

## Installation

```bash
pip install pandas-processors
```

## Features

### 1. Imputation

The library provides several imputation strategies for handling missing values:

#### Mean/Median Imputation

```python
import pandas as pd
from pandas_processors.impute import MeanMedianImputer

# Create a DataFrame with missing values
df = pd.DataFrame({'A': [1, 2, None], 'B': [4, None, 6]})

# Impute missing values with mean
imputer = MeanMedianImputer(columns=['A', 'B'], imputation_method='mean')
imputed_df = imputer.impute(df)
```

#### Constant Imputation

```python
from pandas_processors.impute import ConstantImputer

# Create a DataFrame with missing values
df = pd.DataFrame({'A': [1, 2, None], 'B': [4, None, 6]})

# Impute missing values with a constant value
imputer = ConstantImputer(columns=['A', 'B'], value=0)
imputed_df = imputer.impute(df)
```

#### Categorical Imputation

```python
from pandas_processors.impute import CategoryImputer

df = pd.DataFrame({'A': ['a', 'b', None], 'B': ['x', None, 'z']})

# Impute missing values in categorical columns
imputer = CategoryImputer(
    columns=['A', 'B'],
    imputation_method='missing',
    fill_value='Missing'
)
imputed_df = imputer.impute(df)
```

### 2. Feature Creation

#### Sum Features

```python
from pandas_processors.create import SumFeatures

# Create a new feature by summing existing features
X = pd.DataFrame({
    "col1": [1, 2, 3],
    "col2": [1, 1, 1],
    "col3": [2, 2, 2]
})

sum_feature = SumFeatures(
    features=["col1", "col2", "col3"],
    new_feature_name="col4",
    weights=[1, 2, 0.5]
)
X = sum_feature.fit_transform(X)
```

#### Conditional Features

```python
from pandas_processors.create import ConditionalFeatures

X = pd.DataFrame({"col1": [-1, 2, 3], "col2": [4, -5, 6]})

# Create new features based on conditions
conditional_feature = ConditionalFeatures(
    features=["col1", "col2"],
    new_feature_names=["col3", "col4"],
    condition=lambda x: x > 0,
    true_value=1,
    false_value=0
)
X = conditional_feature.fit_transform(X)
```

### 3. Feature Normalization

#### Skewed Feature Normalization

```python
import numpy as np
from pandas_processors.normalize import SkewedFeatureNormalizer

# Create a DataFrame with skewed features
data = {
    'Feature1': np.random.exponential(scale=2, size=1000),
    'Feature2': np.random.lognormal(mean=1, sigma=0.5, size=1000),
    'Feature3': np.random.gamma(shape=2, scale=1, size=1000)
}
df = pd.DataFrame(data)

# Normalize skewed features
normalizer = SkewedFeatureNormalizer(threshold=0.5)
normalized_df = normalizer.fit_transform(df)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
