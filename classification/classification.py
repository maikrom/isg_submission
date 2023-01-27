import pandas as pd
from autogluon.tabular import TabularPredictor


def pred_to_file(pred, filename):
    predictions = pd.DataFrame(columns=['Id','Predicted'])
    predictions.Id = X_submit.Id
    predictions.Predicted = pred
    predictions.set_index('Id',inplace=True)
    predictions.to_csv(filename)


# Load data
X = pd.read_csv("data/train_features.csv")
y = pd.read_csv("data/train_label.csv")
X_submit = pd.read_csv("data/test_features.csv")

# alignment
df = pd.concat([X, pd.DataFrame(y['label'])], axis=1)
X, y = df.drop('label',axis=1,inplace=False), df['label']

# Load into single dataframe for Autogluon
train_data = pd.concat([X, pd.DataFrame(y)], axis=1)

# Autogluon Predictor
predictor = TabularPredictor(label="label", eval_metric='f1_macro', path='models_class', verbosity=3).fit(
    train_data, presets='best_quality', time_limit=43200, num_gpus=1, num_bag_folds=8, num_bag_sets=20
)

# Print results
results = predictor.fit_summary()
print(train_data.info())

predictor = TabularPredictor.load('models_class')

# The most effective model was the FastAI Neural Net, stacked twice
y_submit = predictor.predict(X_submit, model='NeuralNetFastAI_BAG_L2')
pred_to_file(y_submit, f'NeuralNetFastAI_BAG_L2.csv')

# The Weighted Ensemble L3 was close, but was always outperformed
y_submit = predictor.predict(X_submit, model='WeightedEnsemble_L3')
pred_to_file(y_submit, f'WeightedEnsemble_L3.csv')