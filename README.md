# mlfs

Custom-built Python library showcasing key machine learning algorithms implemented from scratch using NumPy. Designed to demonstrate deep understanding of algorithmic foundations, data preprocessing, and model evaluation. Includes detailed performance comparisons with `scikit-learn` for each method.

---

## 🚀 Features

* **Linear Regression** (`mlfs.linear_regression.LinearRegression`)
* **Logistic Regression** (`mlfs.logistic_regression.LogisticRegression`)
* **K-Means Clustering** (`mlfs.kmeans.KMeans`)
* **K-Nearest Neighbors** (`mlfs.knn.KNN`)
* **Gaussian Naive Bayes** (`mlfs.naive_bayes.NaiveBayes`)
* **Support Vector Machine** (`mlfs.svm.SVM`)
* **Decision Tree** (`mlfs.decision_tree.DecisionTree`)

Each algorithm is implemented in pure Python/NumPy with no heavy dependencies, so you see exactly how they work under the hood.

---

## 📂 Project Structure

```text
mlfs/
├── .gitignore
├── README.md
├── data/
│   ├── .gitkeep
│   ├── iris.csv
│   └── breast-cancer.csv
├── mlfs/
│   ├── __init__.py
│   ├── linear_regression.py
│   ├── logistic_regression.py
│   ├── kmeans.py
│   ├── knn.py
│   ├── naive_bayes.py
│   ├── svm.py
│   ├── decision_tree.py
│   ├── metrics.py
│   └── preprocessing.py
├── notebooks/
│   ├── .gitkeep
│   ├── compare_linear_regression.ipynb
│   ├── compare_logistic_regression.ipynb
│   ├── compare_kmeans.ipynb
│   ├── compare_knn.ipynb
│   ├── compare_naive_bayes.ipynb
│   ├── compare_svm.ipynb
│   └── compare_decision_tree.ipynb
├── requirements.txt
├── setup.py
└── tests/
    ├── .gitkeep
    ├── test_linear_regression.py
    ├── test_logistic_regression.py
    ├── test_kmeans.py
    ├── test_knn.py
    ├── test_naive_bayes.py
    ├── test_svm.py
    └── test_decision_tree.py
```

---

## 📦 Installation

### Prerequisites

* Python ≥ 3.8
* (Optional) A virtual environment tool such as `venv` or `conda`

### From PyPI

```bash
pip install mlfs
```

### From Source

```bash
git clone https://github.com/<your-username>/mlfs.git
cd mlfs
pip install -r requirements.txt
pip install -e .
```

---

## ⚙️ Usage

### 1. Import your algorithm of choice

```python
from mlfs.linear_regression import LinearRegression
from mlfs.naive_bayes import NaiveBayes
```

### 2. Prepare data

```python
from mlfs.preprocessing import train_test_split, standardize

X, y = ...  # load or generate data
X_train, X_test, y_train, y_test = train_test_split(X, y)
X_train, mean, std = standardize(X_train, return_params=True)
X_test = (X_test - mean) / std
```

### 3. Fit and predict

```python
lr = LinearRegression()
lr.fit(X_train, y_train, iterations=1000)
preds = lr.predict(X_test)

nb = NaiveBayes()
nb.fit(X_train, y_train)
preds = nb.predict(X_test)
```

---

## 📊 Comparison Notebooks

Use the comparison notebooks to benchmark your implementations against `scikit-learn`:

* `compare_linear_regression.ipynb`
* `compare_logistic_regression.ipynb`
* `compare_kmeans.ipynb`
* `compare_knn.ipynb`
* `compare_naive_bayes.ipynb`
* `compare_svm.ipynb`
* `compare_decision_tree.ipynb`

Run:

```bash
jupyter notebook compare_<algorithm>.ipynb
```

They include performance metrics, timing, and scalability plots.

---

## 📁 Data

Example datasets are stored in `data/`:

* `breast-cancer.csv`
* `iris.csv`

You can replace them with your own datasets as needed.

---

## ✅ Testing

Run all tests:

```bash
pytest
```

Test coverage includes all algorithm modules and metrics implementations.

---

## 📑 Requirements

All dependencies are listed in `requirements.txt`, including:

* `numpy`, `pandas`
* `scikit-learn`
* `matplotlib`, `seaborn`, `plotly`
* `pytest`, `memory_profiler`

Install:

```bash
pip install -r requirements.txt
```

---

## 🤝 Contributing

1. Fork the repo
2. Create your feature branch
3. Commit your changes
4. Push the branch
5. Open a PR


