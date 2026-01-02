# Spatial Crime Prediction with Machine Learning  
### Asansol–Durgapur Industrial Belt (India)

This repository contains the complete and reproducible Python implementation used in the research paper:

**“Spatial Crime Prediction in the Asansol–Durgapur Industrial Belt Using Interpretable Machine Learning Models.”**

---

## 📌 Overview

The study focuses on predicting police-station–level crime counts in the Asansol–Durgapur industrial region using interpretable ensemble machine learning models.

Key highlights:
- Random Forest used as the primary predictive model
- XGBoost used as a comparative baseline
- Feature importance analysis for interpretability
- 5-fold cross-validation with RMSE as evaluation metric
- Emphasis on industrial intensity as a dominant crime predictor

---

## 📂 Repository Structure

```powershell
Crime-Prediction-with-ML/
│
├── data/
│ └── asansol_crime_final.csv # Final curated dataset
│
├── src/
│ └── crime_prediction_rf_xgb.py # Final experiment script
│
├── figures/
│ ├── Figure_1_Asansol_Durgapur_Map.png
│ ├── Figure_2_Feature_Importance.png
│ └── Figure_3_Methodology_Flow.png
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Requirements

Python 3.9+

Required libraries:
- pandas
- numpy
- scikit-learn
- xgboost
- matplotlib

Install dependencies using:
```bash
pip install -r requirements.txt
```
---

## ▶️ How to Run

Clone the repository:

```bash
git clone https://github.com/Sagnik7255/Crime-Prediction-with-ML.git
cd Crime-Prediction-with-ML
```

Run the main experiment script:

```bash
python src/crime_prediction_rf_xgb.py
```

---

## 📊 Output

The script outputs:

Cross-validated RMSE for Random Forest

Cross-validated RMSE for XGBoost

Feature importance scores from Random Forest

These results are reported and discussed in the accompanying research paper.

---

## 📚 Data Sources

Crime statistics and spatial analysis: IOSR Journal

City-level crime perception indices: Numbeo

Aggregated crime statistics: NCRB (Government of India)

All datasets used are derived from publicly available sources.

---

## 🔬 Reproducibility

All experiments are deterministic where applicable (fixed random seeds).
The codebase is intended to support reproducibility and academic transparency.

---

## 📜 License

This repository is intended for academic and research use.
A license may be added depending on publication requirements.

---

## ✉️ Contact

Author: Sagnik Chakrabarti.
Contact address: csagnik752@gmail.com.
For questions related to the code or study, please open an issue or contact via GitHub.
