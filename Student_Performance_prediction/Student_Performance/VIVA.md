# CA1 Viva Questions and Short Answers

1. **What is the aim?** Predict a numeric Performance Index using study-related factors.
2. **What kind of ML?** Supervised regression.
3. **What is the target?** Performance Index.
4. **What are the predictors?** Hours Studied, Previous Scores, Extracurricular Activities, Sleep Hours, Question Papers Practiced.
5. **Why encode extracurricular activities?** Linear models need numeric input; Yes=1 and No=0.
6. **What is SLR?** Linear regression with one predictor.
7. **What is MLR?** Linear regression with multiple predictors.
8. **What does a coefficient mean?** Expected change in target for a one-unit predictor increase holding others fixed (with scaling, per standard deviation).
9. **How is least squares estimated?** Minimize the sum of squared residuals; matrix solution uses (XᵀX)⁻¹Xᵀy when invertible.
10. **What is a residual?** Actual minus predicted target.
11. **What is polynomial regression?** A linear model on polynomial feature transformations.
12. **What is overfitting?** Fitting training noise so performance worsens on unseen observations.
13. **What is underfitting?** A model is too simple to capture useful patterns.
14. **What is bias–variance tradeoff?** Increasing flexibility may reduce bias but increase prediction variance.
15. **What is Ridge?** Regression with an L2 squared-coefficient penalty.
16. **What is Lasso?** Regression with an L1 absolute-coefficient penalty, often setting some coefficients to zero.
17. **What is alpha?** Strength of the regularization penalty.
18. **Why scale?** Regularization penalties depend on feature scales.
19. **What is cross-validation?** Repeatedly train on subsets and validate on held-out folds.
20. **Why 5-fold?** A practical tradeoff between computational cost and estimate stability.
21. **What is GridSearchCV?** It compares candidate hyperparameters using cross-validation.
22. **Why not tune on the test set?** That leaks information and makes final performance optimistic.
23. **What is MAE?** Mean absolute prediction error.
24. **What is MSE?** Mean squared prediction error.
25. **What is RMSE?** Square root of MSE, in target units.
26. **What is R²?** Improvement in squared error over predicting the test target mean; may be negative.
27. **Which model is best?** State the one with lowest training CV RMSE from your actual run.
28. **What is multicollinearity?** Predictors are strongly correlated, making coefficient estimates unstable.
29. **What assumptions matter?** Approximate linear mean relationship, independent errors, constant error variance and no severe collinearity; normal residuals mainly matter for classical inference.
30. **What is a limitation?** Observational correlations do not prove causes; data may not represent all students.
