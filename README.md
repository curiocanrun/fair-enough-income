<small>

## Fair Adult Income Prediction

Can we predict income without unfairly favoring one gender over another?

We explore this question on the Adult Income dataset. A standard XGBoost model shows clear bias - men are over-predicted as high earners. We try balancing protected groups using SMOTE. It doesn't help much. The bias is not in gender itself but in proxies like occupation, hours, and education.

Finally, we apply a post-processing technique: we adjust decision thresholds separately for each group. This works. Equalized Odds drops below 0.1 while accuracy stays the same.

#### References

[1] Dorleon, G., Megdiche, I., Bricon-Souf, N., & Teste, O. (2023). **FAPFID: A Fairness-Aware Approach for Protected Features and Imbalanced Data.** *Transactions on Large-Scale Data- and Knowledge-Centered Systems.*

[2] Hardt, M., Price, E., & Srebro, N. (2016). **Equality of Opportunity in Supervised Learning.** *Advances in Neural Information Processing Systems (NeurIPS).*

[3] Chen, T., & Guestrin, C. (2016). **XGBoost: A Scalable Tree Boosting System.** *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining.*

For a broader discussion on why accuracy alone is not enough for evaluating classification models, see:

[Beyond Accuracy: Evaluation of Classification Models](https://medium.com/@charithasrig/beyond-accuracy-evaluation-of-classification-models-0ecc00d84257)

</small>
