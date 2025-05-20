An implementation of the Correlation-based Feature Selection (CFS) algorithm in Python.
The implementation uses a simple sklearn API for selection.

Inspirations:
- A blog post on its implementation https://johfischer.com/2021/08/06/correlation-based-feature-selection-in-python-from-scratch/
- The original thesis that describes the algorithm https://www.cs.waikato.ac.nz/~mhall/thesis.pdf by Mark A. Hall
- The Weka implementation of the algorithm, via its Java source code https://git.cms.waikato.ac.nz/weka/weka

# Quickstart

Install with `pip install git+https://...`
It uses my numba implementation of minimum description length (MDL) computation, since the other libraries I had found were too slow for larger datasets. See the dependency list in `pyproject.toml`.

```python
# Load dataset
from sklearn.datasets import fetch_openml
df = fetch_openml(
    name="madelon", version=1, as_frame=True, parser="pandas"
).frame
features = [name for name in df.columns if name != "Class"]
df["Class"] = df["Class"].astype("int") - 1
df[features] = df[features].astype("float")

# Create the selector and use it directly :) 
# it is a sklearn compatile feature selector
# it also works with pandas DataFrames as it should
from correlation_based_feature_selection import CorrelationBasedFeatureSelection
X = CorrelationBasedFeatureSelector().fit_transform(df[features], df[label])
```
