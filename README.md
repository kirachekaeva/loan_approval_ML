# Обучение модели для решения задачи кредитного скоринга
### Основные технологии
| Технология | Назначение |
|------------|------------|
| **Python** | Основной язык разработки |
| **Pandas / NumPy** | Обработка и анализ данных |
| **Scikit-learn** | ML алгоритмы и предобработка |
| **SMOTE** | работа с дисбалансом классов |
| **XGBoost** | Градиентный бустинг |
| **Matplotlib / Seaborn** | Визуализация данных |
### Ключевые методы ML
- **GridSearchCV** — подбор гиперпараметров
- **One-Hot Encoding** — кодирование категориальных признаков
- **StandardScaler** — масштабирование числовых признаков
- **Stratified Sampling** — стратифицированное разделение выборки
## Данные

Датасет `loan_data.csv` (https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data) содержит 45 000 записей с 14 признаками:

### Структура данных
| Признак | Тип | Описание |
|---------|-----|----------|
| `person_age` | numerical | Возраст заемщика |
| `person_gender` | categorical | Пол заемщика |
| `person_education` | categorical | Уровень образования |
| `person_income` | numerical | Годовой доход |
| `person_emp_exp` | numerical | Опыт работы в годах |
| `person_home_ownership` | categorical | Статус владения жильем (аренда, собственность, ипотека) |
| `loan_amnt` | numerical | Запрашиваемая сумма кредита |
| `loan_intent` | categorical | Цель кредита |
| `loan_int_rate` | numerical | Процентная ставка по кредиту |
| `loan_percent_income` | numerical | Сумма кредита в процентах от годового дохода |
| `cb_person_cred_hist_length` | numerical | Длина кредитной истории в годах |
| `credit_score` | numerical | Кредитный скоринг |
| `previous_loan_defaults_on_file` | categorical | Наличие дефолтов по предыдущим кредитам |
| `loan_status` | binary | **Целевая переменная** (1 = одобрен, 0 = отклонен) |
## Результаты проделанной работы
- Проведен полноценный разведочный анализ данных с визуализацией (box-plot, countplot, heatmap, histplot)
- Создан пайплайн для предобработки данных
- Были обучены 4 модели машинного обучения: ```LogisticRegression```, ```DecisionTreeClassifier```, ```RandomForestClassifier```, ```XGBClassifier```
- Для оценки результатов наиболее эффективных моделей были использованы метрики классификации, представленные в отчете о классификации (classification_report) и матрице ошибок (confusion_matrix)
### Сравнение моделей
| Модель | ROC-AUC | Accuracy |
|---------|-----|----------|
| `Logistic Regression` | 0.96 | 0.86 |
| `Decision Tree` | 0.97 | 0.89 |
| `Random Forest` |  0.99 | 0.92 |
| `XGBoost` | 0.98 | 0.94 |