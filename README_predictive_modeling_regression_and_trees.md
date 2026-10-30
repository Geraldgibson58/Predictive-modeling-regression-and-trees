# Predictive Modeling: Used-Car Prices and eBay Auction Competitiveness

## Project Overview
This project applies supervised machine learning and statistical modeling to two prediction problems: estimating the selling prices of used Toyota Corolla vehicles and classifying whether eBay auctions are competitive. It compares multiple linear regression, decision trees, and ensemble classifiers, with particular attention to model accuracy, feature importance, and overfitting.

The analyses were completed in Python as part of an ANA615 data science assignment.

## Project Objectives
- Predict used-car selling prices from vehicle specifications using multiple linear regression.
- Evaluate regression performance on training, validation, and test partitions.
- Classify competitive eBay auctions using a decision tree, AdaBoost, bagging, and random forest.
- Compare classification accuracy and first-decile lift across models.
- Test whether tuning tree complexity improves used-car price predictions.
- Compare continuous price regression with classification into 20 price ranges.

## Datasets

**Toyota Corolla (`ToyotaCorolla.csv`)**  
Contains 1,436 used-car records with price, age, mileage, horsepower, fuel type, and equipment attributes. The notebook uses `Price` as the regression target and a selected set of 15 original predictors (fuel type is dummy-encoded).

**eBay Auctions (`eBayAuctions.csv`)**  
Contains auction characteristics such as product category, currency, seller rating, duration, closing day, and prices. The target is `Competitive?`.

> These are historical instructional datasets, not current market listings. Both CSV files are required to run the complete notebook.

## Methods

### 1. Toyota Corolla: Multiple Linear Regression
- Prepared the dataset and dummy-encoded `Fuel_Type`.
- Split records into 50% training, 30% validation, and 20% test sets.
- Fitted an ordinary least squares (OLS) regression model using `statsmodels`.
- Evaluated performance with root mean squared error (RMSE), mean absolute error (MAE), and R².
- Ranked predictors using absolute standardized regression coefficients.

| Dataset | RMSE | MAE | R² |
|---|---:|---:|---:|
| Training | 1,235.91 | 914.84 | 0.8931 |
| Validation | 1,135.03 | 909.15 | 0.8868 |
| Test | 1,271.88 | 958.77 | 0.8746 |

**Finding:** Vehicle age was the most influential predictor by absolute standardized coefficient, followed by quarterly tax, kilometers driven, and automatic air conditioning. The test-set R² was approximately **0.875**.

### 2. eBay Auctions: Classification and Ensemble Models
- Divided records into 60% training and 40% validation sets.
- One-hot encoded categorical predictors.
- Trained a decision tree, AdaBoost classifier, bagged decision trees, and random forest.
- Compared validation accuracy and **first-decile lift** (the response rate in the top-scored 10% relative to the overall response rate).

| Classifier | Validation accuracy | First-decile lift |
|---|---:|---:|
| Decision tree | 87.58% | 1.7180 |
| AdaBoost | 88.47% | 1.7867 |
| Bagging | 87.58% | 1.8096 |
| Random forest | 86.44% | 1.8096 |

**Finding:** AdaBoost achieved the highest validation accuracy in this comparison. Bagging and random forest tied for the highest first-decile lift.

### 3. Toyota Corolla: Regression Trees and Price-Bin Classification
- Used a 60% training / 40% validation split for regression trees.
- Fitted an unconstrained decision-tree regressor, then tuned tree parameters with five-fold `GridSearchCV`.
- Used a separate classification tree to predict one of 20 price bins.
- Compared feature importance, model errors, and prediction types.

| Regression tree | Training RMSE | Validation RMSE |
|---|---:|---:|
| Full-grown tree | 0.00 | 1,492.34 |
| Tuned tree (max depth 5) | 1,028.03 | 1,152.41 |

**Finding:** The full tree fitted the training data exactly but had larger validation error, illustrating overfitting. Limiting the depth to 5 reduced validation RMSE to **1,152.41**. Age was the dominant feature in the tuned regression tree.

The tuned price-bin classification tree achieved **48.0% validation accuracy** across 20 unevenly populated price categories. This categorical approach is less precise for predicting an exact selling price.

## Tools and Libraries
- Python and Jupyter Notebook
- pandas and NumPy — data preparation
- statsmodels — OLS regression
- scikit-learn — dataset splitting, tree models, ensemble models, metrics, and grid search
- Matplotlib — prediction-error boxplots

## Running the Project
1. Download the notebook and both datasets into the same project folder.
2. Rename the uploaded Toyota dataset from `ToyotaCorolla(3).csv` to `ToyotaCorolla.csv`, or update the notebook's `pd.read_csv` paths accordingly.
3. Install Python dependencies:

   ```bash
   pip install jupyter pandas numpy matplotlib scikit-learn statsmodels
   ```
4. Open the notebook with Jupyter Notebook or JupyterLab.
5. **Before using Run All**, change the assignment prompts and written-answer cells from **Code** to **Markdown**. They are presently stored as code cells containing prose and will raise Python syntax errors.
6. Run the analyses in notebook order.

## Limitations and Future Improvements
- The datasets are historical and are not intended to predict today's prices or auction outcomes without retraining.
- The notebook's model comparison is based on the supplied data partitions; broader cross-validation would improve assessment of model reliability.
- The price-bin classification task has rare and unevenly distributed categories, making classification accuracy harder to interpret.
- Future work could compare more regression algorithms, investigate potential information leakage in auction predictors, and evaluate performance on completely new data.

## Key Takeaway
The project illustrates that model selection depends on the prediction goal and evaluation metric. Linear regression provides interpretable continuous-price predictions; tree tuning helps control overfitting; and ensemble methods can improve classification performance, though the best approach varies by metric.

## Academic Context
ANA615 coursework project. Results and metrics above are taken from the submitted notebook output.
