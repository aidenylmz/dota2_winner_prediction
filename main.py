# %%
from numpy import mean
from numpy import std
import pandas as pd
import numpy as np
import json
import collections
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
# %%
# --------------------------------------------
# Adds header to the data including hero names
# --------------------------------------------
headers = ['winner_team', 'cluster_id', 'game_mode', 'game_type']

# %%
with open('docs/heroes.json') as f:
    heroes_json = json.load(f)

# %%
heroes_list = heroes_json['heroes']

# %%
heroes_dict = {}
for heroes in heroes_list:
    id = heroes['id']
    name = heroes['name']
    heroes_dict[id] = name

sorted_heroes_dict = collections.OrderedDict(sorted(heroes_dict.items()))

for hero in sorted_heroes_dict.values():
    headers.append(hero)
# %%
train_df = pd.read_csv("docs/dota2Train.csv", names=headers, index_col=False)
test_df = pd.read_csv("docs/dota2Test.csv", names=headers, index_col=False)
# %%
# --------------------------------------------
# Correlation between columns
# (Delete these lines if it takes too much time.)
# --------------------------------------------
# plt.figure(figsize=(18, 8))
# sns.heatmap(train_df.corr(), annot=True)
# plt.xticks(rotation=0, fontsize=8)
# plt.yticks(fontsize=8)
# plt.savefig('docs/correlation_heatmap.png')
# %%
# --------------------------------------------
# Separates y value which is winner team and leaves remaining data 'X'
# --------------------------------------------
y = train_df.pop('winner_team')
X = train_df
# %%
# --------------------------------------------
# Plots feature importance with Extra Trees
# --------------------------------------------


def sort_features(feature_importances, columns):
    important_features_dict = {}
    for x, i in enumerate(feature_importances):
        important_features_dict[columns[x]] = i

    important_features_list = sorted(important_features_dict,
                                     key=important_features_dict.get,
                                     reverse=True)

    return important_features_list


def change_width(ax, new_value):
    for patch in ax.patches:
        current_width = patch.get_width()
        diff = current_width - new_value

        # we change the bar width
        patch.set_width(new_value)

        # we recenter the bar
        patch.set_x(patch.get_x() + diff * .5)


# %%
# --------------------------------------------
# Extra Trees on the model
# --------------------------------------------
model = ExtraTreesRegressor()
model.fit(X, y)
# %%
# fig, ax = plt.subplots()
# plt.figure(figsize=(100, 25))

# sorted_features = sort_features(
#     model.feature_importances_, X.columns)

# sns.barplot(ax=ax, x=sorted_features, y=sorted(
#     model.feature_importances_, reverse=True))

# fig.savefig('docs/feature_importance.png')
# %%
# Test set
y_test = test_df.pop('winner_team')
X_test = test_df
# %%
# y_pred = model.predict(X_test)
# # %%
# rsqrd = r2_score(y_test, y_pred)
# # %%
# print(f'R2 Error: {rsqrd}')
# # %%
# scores = cross_val_score(
#     model, X, y, scoring='neg_mean_squared_error', cv=5)
# rmse = np.sqrt(-scores)
# # %%
# print('Reg rmse:', rmse)
# print('Reg Mean:', rmse.mean())
# print('---------------------------------------')

# plt.figure(figsize=(18, 8))
# sns.histplot(y_test - y_pred)
# plt.savefig('docs/distplot.png')

# plt.figure(figsize=(10, 5))
# plt.scatter(y_test, y_pred)
# plt.savefig('docs/scatter.png')

# %%
# evaluate a given model using cross-validation


def evaluate_model(model, X, y):
    # define the evaluation procedure
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    # evaluate the model and collect the results
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    return scores


# %%
scores = evaluate_model(model, X, y)
# summarize the performance along the way
print(f'Score Mean: {mean(scores)}, Score Std.: {std(scores)}')
