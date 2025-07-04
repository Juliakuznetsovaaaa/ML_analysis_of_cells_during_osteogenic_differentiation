import pandas as pd
import numpy as np
import matplotlib
from sklearn.utils import compute_class_weight

matplotlib.use('Agg')  # Важно: используем бэкенд без GUI
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.pipeline import make_pipeline
from collections import Counter
import joblib
import json
import datetime
import os
from scipy.stats import randint, uniform, loguniform
from sklearn.base import clone


def load_data():
    dfs = []
    for file, class_name in [
        ('full_dapi_good_od.csv', 'healthy_osteo'),
        ('full_dapi_good_cntrl.csv', 'healthy_control'),
        ('full_dapi_bad_od.csv', 'disease_osteo'),
        ('full_dapi_bad_cntrl.csv', 'disease_control')
    ]:
        df = pd.read_csv(file)
        df['class'] = class_name
        cols_to_drop = [col for col in ['Label', ' '] if col in df.columns]
        if cols_to_drop:
            df.drop(columns=cols_to_drop, inplace=True)
        dfs.append(df)

    # Конкатенация с оптимизацией типов данных
    data = pd.concat(dfs, ignore_index=True).convert_dtypes()
    return data


data = load_data()
# Удаление нечисловых столбцов кроме целевого
non_numeric_cols = data.select_dtypes(exclude=np.number).columns.difference(['class'])
if not non_numeric_cols.empty:
    print(f"Удаление нечисловых столбцов: {list(non_numeric_cols)}")
    data = data.drop(columns=non_numeric_cols)
data = data.loc[:, ~data.columns.duplicated()]
# Удаление некорректных значений
data = data.replace([np.inf, -np.inf], np.nan).dropna()
# Кодирование целевой переменной
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data['class'])
X = data.drop('class', axis=1)


def select_top_features(X, y, n_features=5):
    """Выбор n наиболее важных признаков с помощью RandomForest"""
    selector = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    selector.fit(X, y)

    # Получение важности признаков
    importances = selector.feature_importances_
    indices = np.argsort(importances)[::-1]

    # Выбор топ-N признаков
    top_features = X.columns[indices[:n_features]]
    print(f"\nТоп-{n_features} важных признаков:")
    for i, feature in enumerate(top_features, 1):
        print(f"{i}. {feature}: {importances[indices[i - 1]]:.4f}")

    return top_features


# Выбор самых важных признаков
top_features = select_top_features(X, y, n_features=5)
X = X[top_features]
print(f"\nВсего признаков: {X.shape[1]}")
print(f"Всего образцов: {X.shape[0]}")
print("Распределение классов:")
class_dist = pd.Series(y).value_counts()
print(class_dist)
print("Кодировка классов:", label_encoder.classes_)

# 3. Стратифицированное разделение с учетом дисбаланса
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nРаспределение классов в обучающей выборке:")
print(pd.Series(y_train).value_counts())


def evaluate_model(model, model_name, X_test, y_test, results_dir, classes):
    """Расширенная оценка модели с дополнительными метриками и визуализациями"""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')

    print(f"\n{model_name} Accuracy: {accuracy:.4f}")
    print(f"{model_name} ROC-AUC: {roc_auc:.4f}")

    report = classification_report(y_test, y_pred, target_names=classes, output_dict=True)
    print("\nОтчет о классификации:")
    print(pd.DataFrame(report).transpose())

    # Сохранение метрик
    metrics = {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'classification_report': report
    }

    # Визуализация матрицы ошибок
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('Истинные метки')
    plt.xlabel('Предсказанные метки')
    plt.tight_layout()
    plt.savefig(f'{results_dir}/{model_name}_confusion_matrix.png')
    plt.close()

    # ROC-кривые
    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(classes):
        RocCurveDisplay.from_predictions(
            y_test == i,
            y_proba[:, i],
            name=f"{class_name} vs Rest",
            ax=plt.gca()
        )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f'ROC Curves - {model_name}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.savefig(f'{results_dir}/{model_name}_roc_curves.png')
    plt.close()

    return metrics


def train_and_evaluate_models(X_train, y_train, X_test, y_test, results_dir, classes):
    """Обучение и оценка моделей с улучшенными пайплайнами"""
    results = {}
    best_models = {}

    # Рассчитаем веса классов для несбалансированных данных
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

    # Конфигурация моделей с пайплайнами
    models = [
        ('rf', make_pipeline(
            StandardScaler(),
            SMOTE(sampling_strategy='auto', random_state=42, k_neighbors=3),
            RandomForestClassifier(random_state=42, class_weight=class_weight_dict)
        ), {
             'randomforestclassifier__n_estimators': randint(200, 800),
             'randomforestclassifier__max_depth': [None, 15, 25, 35],
             'randomforestclassifier__min_samples_split': randint(2, 15),
             'randomforestclassifier__min_samples_leaf': randint(1, 8)
         }),

        ('xgb', make_pipeline(
            StandardScaler(),
            SMOTE(sampling_strategy='auto', random_state=42, k_neighbors=3),
            XGBClassifier(random_state=42, eval_metric='mlogloss',
                          scale_pos_weight=calculate_scale_pos_weight(y_train))
        ), {
             'xgbclassifier__n_estimators': randint(200, 800),
             'xgbclassifier__learning_rate': loguniform(1e-3, 0.3),
             'xgbclassifier__max_depth': randint(5, 15),
             'xgbclassifier__subsample': uniform(0.6, 0.4),
             'xgbclassifier__colsample_bytree': uniform(0.6, 0.4)
         }),

        ('cat', make_pipeline(
            StandardScaler(),
            SMOTE(sampling_strategy='auto', random_state=42, k_neighbors=3),
            CatBoostClassifier(random_state=42, verbose=0, auto_class_weights='Balanced')
        ), {
             'catboostclassifier__iterations': randint(300, 1000),
             'catboostclassifier__learning_rate': loguniform(1e-3, 0.3),
             'catboostclassifier__depth': randint(6, 12),
             'catboostclassifier__l2_leaf_reg': loguniform(1e-3, 10)
         }),

        ('mlp', make_pipeline(
            StandardScaler(),
            SMOTE(sampling_strategy='auto', random_state=42, k_neighbors=3),
            MLPClassifier(random_state=42, max_iter=2000, early_stopping=True)
        ), {
             'mlpclassifier__hidden_layer_sizes': [(100,), (150,), (100, 50), (150, 100), (100, 100)],
             'mlpclassifier__alpha': loguniform(1e-6, 1e-2),
             'mlpclassifier__learning_rate': ['constant', 'adaptive'],
             'mlpclassifier__batch_size': [32, 64, 128]
         })
    ]

    # Обучение моделей
    for name, pipeline, params in models:
        print("\n" + "=" * 60)
        print(f"Обучение модели {name} с подбором гиперпараметров")
        print("=" * 60)

        # Стратифицированная кросс-валидация
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # Увеличили число фолдов

        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=params,
            n_iter=20,
            cv=cv,
            scoring='roc_auc_ovr',
            n_jobs=1,
            verbose=1,
            random_state=42
        )

        search.fit(X_train, y_train)
        best_models[name] = search.best_estimator_

        print(f"Лучшие параметры {name}:")
        print(search.best_params_)

        # Оценка лучшей модели
        metrics = evaluate_model(
            best_models[name],
            name,
            X_test,
            y_test,
            results_dir,
            classes
        )
        results[name] = {
            'accuracy': metrics['accuracy'],
            'roc_auc': metrics['roc_auc'],
            'classification_report': metrics['classification_report'],
            'params': search.best_params_
        }

    print("\n" + "=" * 60)
    print("Обучение модели стекинга")
    print("=" * 60)

    # Выбор лучших моделей для стекинга
    base_models = [
        (f"{name}_best", model)
        for name, model in best_models.items()
    ]

    stacking = StackingClassifier(
        estimators=base_models,
        final_estimator=LogisticRegression(max_iter=5000, class_weight='balanced', random_state=42),
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        n_jobs=1,
        verbose=1
    )

    stacking.fit(X_train, y_train)
    metrics = evaluate_model(stacking, "Stacking", X_test, y_test, results_dir, classes)

    results['stacking'] = {
        'accuracy': metrics['accuracy'],
        'roc_auc': metrics['roc_auc'],
        'classification_report': metrics['classification_report']
    }

    return best_models, stacking, results


# Функция для расчета весов классов
def calculate_scale_pos_weight(y):
    """Рассчитывает веса классов для XGBoost"""
    class_counts = np.bincount(y)
    return class_counts[0] / class_counts[1] if len(class_counts) > 1 else 1


# 8. Создание директории для результатов
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = f"results_{timestamp}"
os.makedirs(results_dir, exist_ok=True)

# 9. Обучение и оценка моделей
trained_models, stacking_model, metrics = train_and_evaluate_models(
    X_train, y_train,
    X_test, y_test,
    results_dir,
    label_encoder.classes_
)

# 10. Сохранение результатов
for name, model in trained_models.items():
    joblib.dump(model, f'{results_dir}/{name}_model.pkl')
joblib.dump(stacking_model, f'{results_dir}/stacking_model.pkl')

# Сохранение метрик и параметров
with open(f'{results_dir}/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=4)

# 11. Финальный отчет
print("\n" + "=" * 60)
print("ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ")
print("=" * 60)
for name, res in metrics.items():
    print(f"{name.upper():<10} | Accuracy: {res['accuracy']:.4f} | ROC-AUC: {res['roc_auc']:.4f}")
print("=" * 60)

