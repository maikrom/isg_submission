import pandas as pd
from autogluon.tabular import TabularPredictor

def pred_to_file(pred, filename):
    predictions = pd.DataFrame(columns=['Id', 'Predicted'])
    predictions.Id = X_submit.Id
    predictions.Predicted = pred
    predictions.set_index('Id',inplace=True)
    predictions.to_csv(filename)


# Load data
X = pd.read_csv("data/train_features.csv")
y = pd.read_csv("data/train_label.csv")
X_submit = pd.read_csv("data/test_features.csv")

# Sadly the only feature engineering that worked effectively and consistently for me
X.drop(['feature_3', 'feature_8', 'feature_25', 'feature_26'], axis=1, inplace=True)

# alignment
df = pd.concat([X, pd.DataFrame(y['label'])], axis=1)
X, y = df.drop('label', axis=1, inplace=False), df['label']

# load data into a single dataframe for AutoGluon
train_data = pd.concat([X, pd.DataFrame(y)], axis=1)

# AutoGluon Predictor for 24 hours, needed less, but training time increased with higher bag folds and
# number of bag sets
predictor = TabularPredictor(label="label", eval_metric='rmse', path='models_reg', verbosity=3).fit(
    train_data, presets='best_quality', time_limit=43200, num_gpus=1, num_bag_folds=10, num_bag_sets=30
)

results = predictor.fit_summary()
print(train_data.info())

predictor = TabularPredictor.load('models_reg')

# The weighted ensemble L3 was the most performant everytime
y_submit = predictor.predict(X_submit, model='WeightedEnsemble_L3')
pred_to_file(y_submit, f'WeightedEnsemble_L3.csv')

# FastAi Neural Net performed very similar, but was always outperformed slightly
y_submit = predictor.predict(X_submit, model='NeuralNetFastAI_BAG_L2')
pred_to_file(y_submit, f'NeuralNetFastAI_BAG_L2.csv')
