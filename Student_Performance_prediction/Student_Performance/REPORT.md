# CA1 MACHINE LEARNING PROJECT REPORT

**Title:** Student Performance Prediction Using Advanced Regression and Model Selection  
**Student Name:** __________________  **Roll Number:** __________________  
**Course:** Machine Learning  **Assessment:** CA1 Individual Project  

## Abstract
This project predicts a student's Performance Index from five educational and lifestyle features. We implement simple linear regression, multiple linear regression, polynomial regression, Ridge regression and Lasso regression. We select the model by five-fold cross-validation on training data, tune regularization and polynomial degree using GridSearchCV, and report accuracy on a held-out test set. The best cross-validation model was **Multiple Linear Regression**, with test RMSE **2.0206** and test R² **0.9890** on the specified random split.

## 1. Problem statement and objective
Build a reproducible supervised regression pipeline to estimate Performance Index. Compare simpler and more flexible models, investigate overfitting, and select hyperparameters without using the test set.

## 2. Dataset
The uploaded Student_Performance.csv contains **10,000 rows** and six columns. The target is Performance Index. Predictors: Hours Studied, Previous Scores, Extracurricular Activities (Yes/No), Sleep Hours, and Sample Question Papers Practiced. The dataset was provided with the uploaded GitHub baseline project. It is used here as an educational dataset; do not interpret the relationships as causal.

## 3. Syllabus mapping
**Unit I:** model selection, learning algorithms, simplicity vs complexity, bias–variance, regularization, hyperparameters, evaluation, 5-fold cross-validation and GridSearchCV.  
**Unit II:** simple and multiple linear regression, regression coefficients, matrix least-squares formulation, regression assumptions, Python implementation, polynomial/nonlinear feature transformation, regression pitfalls, Ridge and Lasso regularization.  
The project discusses assumptions and geometry conceptually; it does not formally prove every assumption or graph the exact coefficient constraint contours.

## 4. Methodology
1. Read the CSV; check shape, missing values and numeric summary. Encode extracurricular activities as No=0 and Yes=1.
2. Separate features and target. Split into 80% training and 20% test using random_state=42.
3. Standardize input features **inside a Pipeline**, so each validation fold learns scaling only from its own training portion.
4. Fit SLR using Hours Studied alone; fit MLR, polynomial features + Ridge, Ridge, and Lasso on all five predictors.
5. Tune polynomial degree (2,3) and regularization alpha by GridSearchCV using five-fold shuffled training CV.
6. Compare candidate models using cross-validation RMSE on the training set, then evaluate each fitted model on the held-out test set for transparency. Select the winner using CV, not test results.
7. Save comparison results, plots, coefficients and selected model.

## 5. Theory and key equations
Simple linear regression: y_hat = b0 + b1*x. Multiple linear regression: y_hat = b0 + sum(bj*xj). Matrix least-squares coefficients are beta_hat=(XᵀX)^(-1)Xᵀy if XᵀX is invertible; practical solvers use numerically stable methods. Polynomial regression adds powers/interactions of predictors and remains linear in its coefficients. Ridge minimizes SSE + alpha*sum(beta_j²); Lasso minimizes SSE + alpha*sum(abs(beta_j)), with exact normalization varying by software. MAE=mean(abs(y-y_hat)); MSE=mean((y-y_hat)²); RMSE=sqrt(MSE); R²=1-SSE/SST. Regularization discourages unstable coefficients; a high polynomial degree may overfit.

## 6. Results (actual execution)
| Model | CV RMSE | Test MAE | Test RMSE | Test R² |
|---|---:|---:|---:|---:|
| Multiple Linear Regression | 2.0432 | 1.6111 | 2.0206 | 0.9890 |
| Ridge Regression | 2.0432 | 1.6111 | 2.0206 | 0.9890 |
| Lasso Regression | 2.0432 | 1.6112 | 2.0207 | 0.9890 |
| Polynomial Regression | 2.0447 | 1.6115 | 2.0201 | 0.9890 |
| Simple Linear Regression | 17.7919 | 15.5339 | 17.9392 | 0.1316 |

**Best selected model:** Multiple Linear Regression (lowest 5-fold CV RMSE: 2.0432).  
**Hyperparameters:** `{"Polynomial Regression": {"model__alpha": 0.1, "poly__degree": 2}, "Ridge Regression": {"model__alpha": 0.01}, "Lasso Regression": {"model__alpha": 0.001}}`.

The small CV RMSE differences among multiple linear, Ridge, and Lasso indicate that a simple linear relationship is already strong for this dataset. The SLR baseline performs substantially worse because it omits other predictive variables. The held-out test results are reported only after selection.

## 7. Charts and interpretation
- `results/cv_model_comparison.png`: compares training CV RMSE; smaller is better.
- `results/actual_vs_predicted.png`: checks whether predictions lie near the ideal diagonal.
- `results/residual_plot.png`: checks whether residuals show patterns or changing spread.
- `results/bias_variance_demo.png`: compares training and validation errors for polynomial degrees 1–4, illustrating complexity effects. It is a diagnostic demonstration rather than a causal proof of bias/variance.
- `results/linear_coefficients.csv`: coefficients from the standardized multiple regression model.

## 8. Regression assumptions and limitations
Inspect linearity using residuals, independence by study design, homoscedasticity by residual spread, and multicollinearity by feature correlation or VIF if required. Residual normality is mainly important for conventional inference, not necessarily for predictive performance. A high score on one dataset does not guarantee external validity. Student behavior and educational outcomes can be affected by unmeasured factors; do not use the model for consequential individual decisions.

## 9. Conclusion
The project implements and evaluates five regression approaches while covering both syllabus units. The best CV model is Multiple Linear Regression; it achieved a held-out test RMSE of 2.0206 and R² of 0.9890. Cross-validation and training-only hyperparameter selection reduce the risk of choosing a model because it happens to perform well on the test set. A future extension could evaluate the model on students from a different school or time period.

## 10. How to execute
`pip install -r requirements.txt` then `python project.py`, or open `CA1_Student_Performance.ipynb` and Run All. Read outputs from `results/`.

## 11. References
- Student-supplied GitHub project: https://github.com/akankshmakam/Student-Performance-Prediction-using-Linear-Regression
- Scikit-learn user guide: https://scikit-learn.org/stable/user_guide.html
- Scikit-learn model selection documentation: https://scikit-learn.org/stable/model_selection.html

**CA1 rubric:** Report 10 marks, Execution 10 marks, Viva 10 marks (30 total). Add your name and ensure you can explain and reproduce the code before submission.
