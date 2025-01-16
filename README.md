# **Malicious URL Detection**

This project aims to detect malicious URLs using advanced machine learning techniques. By extracting key features from raw URL data, applying preprocessing strategies, and leveraging ensemble learning models, the project achieves high accuracy in predicting URL categories (e.g., benign, malicious).

---

## **Key Highlights**

- **Feature Engineering**: Extracted critical features such as URL length, number of dots, dashes, special characters, subdomain levels, query length, and domain patterns to enhance the model's ability to distinguish between benign and malicious URLs.  
- **Data Preprocessing**:  
  - Addressed class imbalance in the dataset using **SMOTE (Synthetic Minority Oversampling Technique)** to generate synthetic samples for minority classes.  
  - Scaled the features using **MinMaxScaler** to ensure uniformity and improve model performance.  

---

## **Modeling and Evaluation**

### Random Forest Classifier  
- Achieved **93.6% accuracy** on the test dataset.  
- Provided a baseline for comparison and demonstrated robust performance due to its ability to handle overfitting and ensemble nature.  

### Stacking Classifier  
- Built a stacking model using **Random Forest** and **XGBoost** as base estimators, with **Logistic Regression** as the meta-classifier.  
- Achieved **95.4% accuracy**, outperforming the standalone Random Forest model by integrating predictions from multiple base models to enhance generalization.  

#### What is Stacking?  
Stacking is an ensemble learning technique that combines multiple models (base estimators) to make predictions. It uses another model (meta-classifier) to learn from the outputs of the base models, improving overall accuracy and reducing bias/variance trade-offs.

---

## **Project Workflow**

1. **Feature Extraction**:  
   - Parsed URLs to derive meaningful features like `URL length`, `number of special characters`, `query parameters`, and `subdomain levels`.  
   - Identified patterns such as `presence of IP in the domain` and `HTTPS usage` to improve malicious detection.  

2. **Data Balancing and Scaling**:  
   - SMOTE was applied to address class imbalance, ensuring equal representation of target classes.  
   - Features were normalized using MinMaxScaler to improve convergence during training.  

3. **Model Training and Optimization**:  
   - Hyperparameter tuning of the Random Forest model was performed using **GridSearchCV** for optimal performance.  
   - Compared the performance of Random Forest and Stacking models based on evaluation metrics like accuracy, F1-score, and ROC-AUC.  

4. **Performance Metrics**:  
   - Random Forest Accuracy: **93.6%**  
   - Stacking Classifier Accuracy: **95.4%**  
   - Demonstrated improved precision, recall, and F1 scores with the Stacking model, highlighting the benefits of ensemble learning.

---

## **Technologies Used**

- **Programming Language**: Python  
- **Libraries**: Pandas, NumPy, Scikit-learn, Imbalanced-learn, XGBoost, Matplotlib, Seaborn  

---

## **Results**

| Model               | Accuracy | F1-Score |  
|---------------------|----------|----------| 
| Random Forest       | 93.6%    | 93.4%    | 
| Stacking Classifier | 95.4%    | 95.3%    |  

The stacking classifier demonstrated significant improvements by leveraging predictions from multiple models.

---

This project showcases the importance of feature engineering, preprocessing, and ensemble learning in building a robust URL classification model.
