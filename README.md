# Machine-Learning-Models-for-HR-Data
Modeling Employee Attrition using Machine Learning Techniques

## Employee Attrition Prediction

This GitHub project predicts employee attrition using the IBM HR Analytics Employee Attrition & Performance Dataset from Kaggle. Implemented entirely in Python, the project leverages Pandas for data manipulation, Pycaret for streamlined machine learning, and Jupyter Notebooks for interactive analysis.

### Tools and Concepts:

1. **Python & Pandas:**
   - Data loading and manipulation.

2. **Pycaret:**
   - Automated model selection and comparison.

3. **Jupyter Notebooks:**
   - Interactive analysis environment.

4. **Binary Classification:**
   - Predicting 'Leaving' (1) or 'Staying' (0).

5. **Linear Discriminant Analysis (LDA):**
   - Chosen classification model.

6. **Reproducibility:**
   - Fixed random state for result consistency.

7. **GitHub:**
   - Version control and collaboration.

### Code Snippets:

```python
# Data loading and splitting
import pandas as pd
dataset = pd.read_csv("attrition_data.csv")
data = dataset.sample(frac=0.95, random_state=786)
data_unseen = dataset.drop(data.index)
data.reset_index(inplace=True, drop=True)
data_unseen.reset_index(inplace=True, drop=True)

# Pycaret setup and model selection
from pycaret.classification import *
exp_clf101 = setup(data=data, target="Attrition", session_id=123)
best_model = compare_models()

# LDA model and visualization
model = create_model('lda')
plot_model(model, plot='feature')
plot_model(model, plot='confusion_matrix')

# Model evaluation on test/validation set
unseen_predictions = predict_model(model, data=data_unseen)
accuracy = check_metric(unseen_predictions['Attrition'], unseen_predictions['Label'], metric='Accuracy')
print(f"Model Accuracy on Unseen Data: {accuracy}")
```

### Graphs:

1. **Feature Importance:**
   - ![Feature Importance](url_to_feature_importance_plot)

2. **Confusion Matrix:**
   - ![Confusion Matrix](https://github.com/vennisabarfi/Machine-Learning-Models-for-HR-Data/blob/main/confusion_matrix.png)

### Getting Started:

Clone or fork this repository to replicate the environment. Follow the README for installation instructions. Contribute to enhance the predictive model for employee attrition in the HR domain.
