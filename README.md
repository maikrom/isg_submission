# isg_submission
My final submission for the ISG Kaggle competition

During the ISG competition I used many different models and different feature engineering techniques.

But during the last two weeks when the AutoGluon Baseline got released my strategy switched very quickly. I hadn't heard of AutoGluon prior to the competition and when I tried it out for the first time. The models that were at that point my best models. (XGBoost, Random Forest, KNN and SVC ensembled and stacked together) were beaten in about 6 hours compute time on my PC by AutoGluon without anu feature engineering. Although my previous ensemble used various Scalers, Outlier removal via IQR and feature selection in the case of the Regression task.

So during the last two weeks I tried to train various AutoGluon models leading to my models folder growing to >150GB. Sadly all the previous scaling and outlier removal techniques either had no improvement upon performance or even made it worse in some cases. Trying different Scalers and outlier removal techniques didn't seem to work either.

So my best bet was to optimize AutoGluon's Stack height and Bagging hyperparameters.

At the point of submitting my last models this seems at least pretty effective, giving me second place on the leaderboard in both public leaderboards. But only the future will tell how those models generalize to the other half of the data.

Luckily this makes running the code yourself very easy, just install the requirements via the below commands and run the python files in the respective folders. (This worked on my Mac and Windows systems). In case this doesn't work you only need to follow the AutoGluon Installation guide and install pandas.
