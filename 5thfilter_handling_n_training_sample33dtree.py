import multiprocessing
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.base import clone
from itertools import product
from joblib import Parallel, delayed

print(f"🧠 Logical CPU cores available: {multiprocessing.cpu_count()}")

#%% Load data
print("\n📂 Loading data...")
data_dir = r'C:\DATABASES_NICOCASTRO\current'
input_pkl_path_v1 = os.path.join(data_dir, 'input_nn_ver1.pkl')
df = pd.read_pickle(input_pkl_path_v1)
print(f"✅ Data loaded. Shape: {df.shape}")
print(f"📁 Columns: {df.columns.tolist()}")

#%% Add noise
def add_noise(tsky, trx=75.0, snr=1000.0):
    tsys = trx + tsky
    tnoise = tsys / snr
    tcal = tsky + tnoise * np.random.standard_normal(tsky.shape)
    return tcal

print("\n🖉 Adding noise to spectra...")
df['SPECTRA_Tb[K]_NOISY'] = df['SPECTRA_Tb[K]'].apply(add_noise)

#%% Prepare data
print("\n📊 Preparing input and output arrays...")
X = np.array(df['SPECTRA_Tb[K]_NOISY'].tolist())
PL = np.array(df['PL[mbar]'].tolist())
H2O = np.array(df['H2O[vmr]'].tolist())
Y = np.concatenate([H2O, PL], axis=1)

#%% Split
print("\n✂️ Splitting into train/val/test...")
X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.4, random_state=42)
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

#%% Normalize
print("\n⚖️ Normalizing...")
scaler_x = StandardScaler()
X_train = scaler_x.fit_transform(X_train)
X_val = scaler_x.transform(X_val)
X_test = scaler_x.transform(X_test)

scaler_y = StandardScaler()
Y_train = scaler_y.fit_transform(Y_train)
Y_val = scaler_y.transform(Y_val)
Y_test = scaler_y.transform(Y_test)

#%% Manual Grid Search
print("\n🌲 Starting manual grid search...")
param_grid = {
    'max_depth': [6, 7, 8],
    'min_samples_split': [2, 10, 12],
    'min_samples_leaf': [5, 6, 7, 8]
}

cv = KFold(n_splits=3, shuffle=True, random_state=42)
param_names = list(param_grid.keys())
param_values = [param_grid[k] for k in param_names]
all_combos = list(product(*param_values))

def evaluate_combination(combo_index, values):
    params = dict(zip(param_names, values))
    print(f"\n [{combo_index + 1}/{len(all_combos)}] Trying: {params}")
    model = MultiOutputRegressor(DecisionTreeRegressor(**params, random_state=42))
    fold_scores = []
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train)):
        model_clone = clone(model)
        model_clone.fit(X_train[train_idx], Y_train[train_idx])
        preds = model_clone.predict(X_train[val_idx])
        rmse = np.sqrt(mean_squared_error(Y_train[val_idx], preds))
        fold_scores.append(rmse)
        print(f"    Fold {fold_idx + 1} RMSE: {rmse:.4f}")
    mean_rmse = np.mean(fold_scores)
    print(f"    Mean CV RMSE: {mean_rmse:.4f}")
    return (params, mean_rmse)

n_jobs_parallel = max(1, multiprocessing.cpu_count() - 4)
print(f"🧵 Using {n_jobs_parallel} CPU cores...")

results = Parallel(n_jobs=n_jobs_parallel)(
    delayed(evaluate_combination)(i, values) for i, values in enumerate(all_combos)
)

results.sort(key=lambda x: x[1])

#%% Final model selection using validation set
top_n = 3
top_candidates = results[:top_n]
val_scores = []

print("\n🔎 Evaluating top models on validation set...")
for idx, (params, _) in enumerate(top_candidates):
    print(f"\n🔁 [{idx+1}/{top_n}] Params: {params}")
    model = MultiOutputRegressor(DecisionTreeRegressor(**params, random_state=42))
    model.fit(X_train, Y_train)
    preds_val = model.predict(X_val)
    rmse_val = np.sqrt(mean_squared_error(scaler_y.inverse_transform(Y_val),
                                          scaler_y.inverse_transform(preds_val)))
    print(f"   🔢 Validation RMSE: {rmse_val:.4f}")
    val_scores.append((model, params, rmse_val))

val_scores.sort(key=lambda x: x[2])
final_model, final_params, final_val_rmse = val_scores[0]

print("\n🏆 Final model selected:")
print(f"   🔧 Params: {final_params}")
print(f"   📊 Validation RMSE: {final_val_rmse:.4f}")

#%% Evaluation on test set
print("\n🧪 Evaluating on test set...")
y_pred_test = final_model.predict(X_test)
y_pred_test = scaler_y.inverse_transform(y_pred_test)
y_true_test = scaler_y.inverse_transform(Y_test)

n_levels = 67
y_true_h2o = y_true_test[:, :n_levels]
y_pred_h2o = y_pred_test[:, :n_levels]
y_true_pl = y_true_test[:, n_levels:]
y_pred_pl = y_pred_test[:, n_levels:]

test_mae = mean_absolute_error(y_true_test, y_pred_test)
test_rmse = np.sqrt(mean_squared_error(y_true_test, y_pred_test))
print(f"\n📊 Test MAE:  {test_mae:.4f}")
print(f"📊 Test RMSE: {test_rmse:.4f}")

mae_h2o = mean_absolute_error(y_true_h2o, y_pred_h2o)
rmse_h2o = np.sqrt(mean_squared_error(y_true_h2o, y_pred_h2o))
mae_pl = mean_absolute_error(y_true_pl, y_pred_pl)
rmse_pl = np.sqrt(mean_squared_error(y_true_pl, y_pred_pl))
print(f"\n🌊 H2O  → MAE: {mae_h2o:.4f} | RMSE: {rmse_h2o:.4f}")
print(f"📉 PL   → MAE: {mae_pl:.4f} | RMSE: {rmse_pl:.4f}")

#%% Plot output index
output_index = 120
print(f"\n📈 Plotting output #{output_index}")
plt.figure(figsize=(10, 5))
plt.plot(y_true_test[:, output_index], label='True', alpha=0.7)
plt.plot(y_pred_test[:, output_index], label='Predicted', alpha=0.7)
plt.title(f"True vs Predicted - Output #{output_index}")
plt.xlabel("Sample index")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#%% Plot vertical profiles
for i in range(3):
    print(f"\n🔎 Test sample #{i}")
    plt.figure(figsize=(7, 6))
    plt.plot(h2o_profile_true := y_true_h2o[i], pl_profile_true := y_true_pl[i], label="True", marker='o')
    plt.plot(h2o_profile_pred := y_pred_h2o[i], pl_profile_pred := y_pred_pl[i], label="Predicted", marker='x')
    plt.gca().invert_yaxis()
    plt.xlabel("H2O content [vmr]")
    plt.ylabel("Pressure Layer [mbar]")
    plt.title(f"H2O vs PL — Test Sample #{i}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

#%%
import seaborn as sns
import matplotlib.pyplot as plt

# Collect all CV results into a DataFrame
results_df = pd.DataFrame([
    {**params, 'cv_rmse': rmse}
    for params, rmse in results
])

# Plot one heatmap per min_samples_split value
for mss in sorted(results_df['min_samples_split'].unique()):
    pivot_table = results_df[results_df['min_samples_split'] == mss].pivot_table(
        index='min_samples_leaf',
        columns='max_depth',
        values='cv_rmse'
    )
    
    plt.figure(figsize=(7, 6))
    sns.heatmap(pivot_table, annot=True, fmt=".3f", cmap='viridis', cbar_kws={'label': 'CV RMSE'})
    plt.title(f"Heatmap of CV RMSE — min_samples_split = {mss}")
    plt.xlabel("max_depth")
    plt.ylabel("min_samples_leaf")
    plt.tight_layout()
    plt.show()

#%%
# Output directory
save_dir = r'C:\DATABASES_NICOCASTRO\AM_software\am_sims\figuras'
os.makedirs(save_dir, exist_ok=True)

# Extract final model hyperparameters
MD = final_params['max_depth']
MSS = final_params['min_samples_split']
MSL = final_params['min_samples_leaf']

h2o_min, h2o_max = 0, 0.04       
pl_min, pl_max = 0, 1100         

# Loop through test samples and save each plot
for i in range(len(y_true_h2o)):
    plt.figure(figsize=(7, 6))
    plt.plot(y_true_h2o[i], y_true_pl[i], label="True", marker='o')
    plt.plot(y_pred_h2o[i], y_pred_pl[i], label="Predicted", marker='x')

    # Fix axes
    #plt.xlim(h2o_min, h2o_max)
    #plt.ylim(pl_max, pl_min)  # Inverted axis (pressure decreases upward)

    # Labels and formatting
    plt.xlabel("H2O content [vmr]")
    plt.ylabel("Pressure Layer [mbar]")
    plt.title(f"H2O vs PL — Test Sample #{i}")
    plt.gca().invert_yaxis()
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Filename with params
    filename = f"TS{i}_MD{final_params['max_depth']}_MSS{final_params['min_samples_split']}_MSL{final_params['min_samples_leaf']}.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath)
    plt.close()
#%%
from sklearn.metrics import r2_score

r2 = r2_score(y_true_test, y_pred_test)
print(f"📈 R² Score (total): {r2:.4f}")

r2_h2o = r2_score(y_true_h2o, y_pred_h2o)
r2_pl  = r2_score(y_true_pl, y_pred_pl)

print(f"🌊 H2O → R²: {r2_h2o:.4f}")
print(f"📉 PL  → R²: {r2_pl:.4f}")
#%%
from joblib import dump, load

final_model, final_params, final_val_rmse = val_scores[0]
model_path = r"C:\DATABASES_NICOCASTRO\AM_software\am_sims\final_model_dt.joblib"
dump(final_model, model_path)
print(f"✅ Final model saved to: {model_path}")
dump(scaler_x, r"C:\DATABASES_NICOCASTRO\AM_software\am_sims\scaler_x.joblib")
dump(scaler_y, r"C:\DATABASES_NICOCASTRO\AM_software\am_sims\scaler_y.joblib")
