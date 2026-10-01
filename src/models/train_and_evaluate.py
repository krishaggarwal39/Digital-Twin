import json
import os
import sys
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    roc_auc_score, precision_score, recall_score, f1_score
)
import xgboost as xgb


def load_data(data_dir="data/processed"):
    """Load train/val/test splits. Returns dict with keys 'train', 'val', 'test',
    each containing 'X_static', 'X_dynamic', 'X_all', 'y_reg', 'y_cls', 'patient_ids'."""
    feature_names_path = os.path.join(data_dir, "feature_names.json")
    with open(feature_names_path, "r") as f:
        feature_names = json.load(f)

    static_cols = feature_names["static"]
    dynamic_cols = feature_names["dynamic"]
    all_cols = feature_names["all"]

    data = {}
    for split in ["train", "val", "test"]:
        csv_path = os.path.join(data_dir, f"features_{split}.csv")
        df = pd.read_csv(csv_path)

        # Replace any inf/nan with 0
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        data[split] = {
            "X_static": df[static_cols].values.astype(np.float32),
            "X_dynamic": df[dynamic_cols].values.astype(np.float32),
            "X_all": df[all_cols].values.astype(np.float32),
            "y_reg": df["target_glucose_2h"].values.astype(np.float32),
            "y_cls": df["target_adverse"].values.astype(np.int32),
            "patient_ids": df["patient_id"].values
        }

    return data, feature_names


def train_models(data):
    """Train 3 XGBoost models for ablation study. Returns dict of models."""
    models = {}
    configs = {
        "static_only": {"X_key": "X_static", "desc": "Static features only (18)"},
        "dynamic_only": {"X_key": "X_dynamic", "desc": "Dynamic features only (29)"},
        "fused": {"X_key": "X_all", "desc": "All features fused (47)"},
    }

    for name, config in configs.items():
        print(f"\n{'='*60}")
        print(f"Training: {name} — {config['desc']}")
        print(f"{'='*60}")

        X_train = data["train"][config["X_key"]]
        y_train = data["train"]["y_reg"]
        X_val = data["val"][config["X_key"]]
        y_val = data["val"]["y_reg"]

        model = xgb.XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=10,
            reg_alpha=0.1,
            reg_lambda=1.0,
            early_stopping_rounds=50,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=50  # Print every 50 rounds
        )

        models[name] = model
        best_iter = getattr(model, "best_iteration", model.n_estimators)
        print(f"Best iteration: {best_iter}")

    return models


def train_classifier(data):
    """Train XGBoost classifier for adverse event prediction."""
    X_train = data["train"]["X_all"]
    y_train = data["train"]["y_cls"]
    X_val = data["val"]["X_all"]
    y_val = data["val"]["y_cls"]

    # Calculate class weight for imbalanced data
    n_pos = np.sum(y_train == 1)
    n_neg = np.sum(y_train == 0)
    scale_pos_weight = float(n_neg / max(n_pos, 1))

    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,
        scale_pos_weight=scale_pos_weight,
        reg_alpha=0.1,
        reg_lambda=1.0,
        early_stopping_rounds=50,
        random_state=42,
        n_jobs=-1,
        eval_metric="auc",
        verbosity=0
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50
    )

    best_iter = getattr(model, "best_iteration", model.n_estimators)
    print(f"Best iteration: {best_iter}")
    return model


def evaluate_models(models, classifier, data, feature_names):
    """Evaluate all models on test set. Returns results dict."""
    results = {}

    configs = {
        "static_only": {"X_key": "X_static", "features": feature_names["static"]},
        "dynamic_only": {"X_key": "X_dynamic", "features": feature_names["dynamic"]},
        "fused": {"X_key": "X_all", "features": feature_names["all"]},
    }

    X_test_all = data["test"]["X_all"]
    y_test_reg = data["test"]["y_reg"]
    y_test_cls = data["test"]["y_cls"]

    def adverse_risk_score(glucose_pred: np.ndarray) -> np.ndarray:
        """Map a predicted glucose value to a monotic adverse-event risk score.

        An adverse event is glucose > 250 (hyper) OR < 54 (hypo), so risk must
        increase as the prediction moves ABOVE the hyper threshold or BELOW the
        hypo threshold. Using the raw prediction as the score (as a naive
        implementation would) is incorrect for hypoglycemia, where a *lower*
        glucose means *higher* risk. We take the distance past the nearer danger
        threshold, which is a valid ranking score for ROC AUC.
        """
        hyper = glucose_pred - 250.0   # positive when above hyper threshold
        hypo = 54.0 - glucose_pred     # positive when below hypo threshold
        return np.maximum(hyper, hypo)

    for name, config in configs.items():
        X_test = data["test"][config["X_key"]]
        model = models[name]

        y_pred = model.predict(X_test)

        # Regression metrics
        rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred))
        mae = mean_absolute_error(y_test_reg, y_pred)
        r2 = r2_score(y_test_reg, y_pred)

        # Derive classification from regression predictions
        y_pred_cls = ((y_pred > 250.0) | (y_pred < 54.0)).astype(int)

        # Classification metrics (only if there are positive cases)
        if np.sum(y_test_cls) > 0 and np.sum(y_pred_cls) > 0:
            precision = precision_score(y_test_cls, y_pred_cls, zero_division=0)
            recall = recall_score(y_test_cls, y_pred_cls, zero_division=0)
            f1 = f1_score(y_test_cls, y_pred_cls, zero_division=0)
        else:
            precision = recall = f1 = 0.0

        # AUROC using a correct symmetric distance-to-danger risk score so that
        # both hyperglycemic and hypoglycemic excursions are ranked properly.
        if len(np.unique(y_test_cls)) > 1:
            auroc = roc_auc_score(y_test_cls, adverse_risk_score(y_pred))
        else:
            auroc = 0.0

        results[name] = {
            "rmse": round(float(rmse), 2),
            "mae": round(float(mae), 2),
            "r2": round(float(r2), 4),
            "auroc": round(float(auroc), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
            "n_features": len(config["features"])
        }

        print(f"\n{name}:")
        for metric, value in results[name].items():
            print(f"  {metric}: {value}")

    # Classifier evaluation
    y_pred_prob = classifier.predict_proba(X_test_all)[:, 1]
    y_pred_cls = classifier.predict(X_test_all)

    if len(np.unique(y_test_cls)) > 1:
        cls_auroc = roc_auc_score(y_test_cls, y_pred_prob)
    else:
        cls_auroc = 0.0

    results["classifier"] = {
        "auroc": round(float(cls_auroc), 4),
        "precision": round(float(precision_score(y_test_cls, y_pred_cls, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test_cls, y_pred_cls, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test_cls, y_pred_cls, zero_division=0)), 4),
    }

    return results


def compute_shap(model, X_test, feature_names_list, output_dir, n_samples=500, static_features_set=None):
    """Compute SHAP values for the fused model on a sample of test data."""
    if static_features_set is None:
        static_features_set = set()

    # Sample test data (SHAP is slow on large datasets)
    n = min(n_samples, len(X_test))
    indices = np.random.RandomState(42).choice(len(X_test), n, replace=False)
    X_sample = X_test[indices]

    # Compute SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Save SHAP values
    np.savez(
        os.path.join(output_dir, "shap_values_sample.npz"),
        shap_values=shap_values,
        X_sample=X_sample,
        feature_names=np.array(feature_names_list)
    )

    # Global feature importance (mean absolute SHAP)
    importance = pd.DataFrame({
        "feature": feature_names_list,
        "mean_abs_shap": np.mean(np.abs(shap_values), axis=0),
        "stream": ["static" if f in static_features_set else "dynamic" for f in feature_names_list]
    }).sort_values("mean_abs_shap", ascending=False)

    importance.to_csv(os.path.join(output_dir, "shap_feature_importance.csv"), index=False)

    print("\nTop 10 features by SHAP importance:")
    print(importance.head(10).to_string(index=False))

    # Create SHAP summary plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names_list, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_summary_plot.png"), dpi=150)
    plt.close()
    print(f"SHAP summary plot saved to {output_dir}/shap_summary_plot.png")

    return shap_values, importance


def evaluate_persistence_baseline(data, feature_names):
    """Naive persistence baseline: predict glucose(t+2h) = glucose(t) = cgm_current.

    This is the shortcut any trivial model could exploit; beating it is the bar
    that proves the model learns real multi-hour physiological dynamics rather
    than just echoing the latest reading.
    """
    all_cols = feature_names["all"]
    cgm_idx = all_cols.index("cgm_current")

    X_test = data["test"]["X_all"]
    y_test_reg = data["test"]["y_reg"]
    y_test_cls = data["test"]["y_cls"]

    y_pred = X_test[:, cgm_idx]  # persistence: future == current

    rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred))
    mae = mean_absolute_error(y_test_reg, y_pred)
    r2 = r2_score(y_test_reg, y_pred)

    return {
        "rmse": round(float(rmse), 2),
        "mae": round(float(mae), 2),
        "r2": round(float(r2), 4),
        "n_features": 1,
        "note": "Naive persistence baseline: predicts glucose(t+2h) = cgm_current",
    }


def main():
    output_dir = "data/processed"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load data
    print("Loading data...")
    data, feature_names = load_data(output_dir)
    for split in ["train", "val", "test"]:
        print(f"  {split}: {len(data[split]['y_reg'])} samples, adverse rate: {data[split]['y_cls'].mean()*100:.1f}%")

    # 2. Train models
    print("\n" + "="*60)
    print("TRAINING ABLATION MODELS")
    print("="*60)
    models = train_models(data)

    # 3. Train classifier
    print("\n" + "="*60)
    print("TRAINING ADVERSE EVENT CLASSIFIER")
    print("="*60)
    classifier = train_classifier(data)

    # 4. Evaluate
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    results = evaluate_models(models, classifier, data, feature_names)

    # 4b. Naive persistence baseline (reproducible, computed from code)
    persistence = evaluate_persistence_baseline(data, feature_names)
    results["persistence_baseline"] = persistence
    print("\npersistence_baseline:")
    for metric, value in persistence.items():
        print(f"  {metric}: {value}")

    # 5. Save models
    models["fused"].save_model(os.path.join(output_dir, "xgb_fused_regressor.json"))
    classifier.save_model(os.path.join(output_dir, "xgb_fused_classifier.json"))
    print(f"\nModels saved to {output_dir}/")

    # 6. Save results
    with open(os.path.join(output_dir, "evaluation_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_dir}/evaluation_results.json")

    # 7. SHAP analysis
    print("\n" + "="*60)
    print("SHAP EXPLAINABILITY ANALYSIS")
    print("="*60)

    static_features_set = set(feature_names["static"])
    shap_values, importance = compute_shap(
        models["fused"],
        data["test"]["X_all"],
        feature_names["all"],
        output_dir,
        static_features_set=static_features_set
    )

    # 8. Print final comparison table
    print("\n" + "="*60)
    print("FINAL ABLATION COMPARISON")
    print("="*60)
    print(f"\n{'Model':<22} {'Features':>9} {'RMSE':>9} {'MAE':>9} {'R²':>9} {'AUROC':>9}")
    print("-" * 72)
    pb = results["persistence_baseline"]
    print(f"{'persistence_baseline':<22} {pb['n_features']:>9} {pb['rmse']:>9.2f} {pb['mae']:>9.2f} {pb['r2']:>9.4f} {'—':>9}")
    for name in ["static_only", "dynamic_only", "fused"]:
        r = results[name]
        print(f"{name:<22} {r['n_features']:>9} {r['rmse']:>9.2f} {r['mae']:>9.2f} {r['r2']:>9.4f} {r['auroc']:>9.4f}")

    print("\n✅ Training and evaluation complete!")


if __name__ == "__main__":
    main()
