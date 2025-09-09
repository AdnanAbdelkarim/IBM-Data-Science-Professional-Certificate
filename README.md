# 🚀 IBM Applied Data Science Capstone (SpaceX)

This branch contains my work for the **IBM Applied Data Science Capstone** (Coursera).  
It demonstrates the full **data science lifecycle**: data collection, SQL exploration, EDA, interactive visualizations, predictive modeling, and business insights.

The project analyzes **SpaceX Falcon 9 launches** to estimate the likelihood of first‑stage landing success and help a competing company make informed bid decisions.

---

## 📂 Project Structure

```
applied-data-science-capstone/
│
├── data/                      # Small, versioned datasets (CSV/SQLite)
│   ├── spacex_cleaned.csv
│   └── spacex.sqlite
│
├── notebooks/                 # Jupyter Notebooks
│   ├── SQL_Commands.ipynb
│   └── IBM_Capstone_Final_Report.ipynb   # Executive Summary, Results & Insights, Conclusion
│
├── scripts/                   # Reusable Python scripts
│   └── spacex_dash_app.py     # Optional Dash demo (local)
│
├── images/                    # Static images used in README / notebook
│   ├── folium_map.png
│   └── dashboard_scatter.png
│
├── requirements.txt
└── README.md
```

> 🔎 If you’re browsing the code here on GitHub: open `notebooks/IBM_Capstone_Final_Report.ipynb` for the polished, end‑to‑end narrative.

---

## 📊 Dataset

- **Source**: SpaceX launch records from the IBM course dataset
- **Files**: `data/spacex_cleaned.csv`, `data/spacex.sqlite`
- **Fields**: launch site, booster version, payload mass (kg), orbit, date, outcome (`Class`)

---

## 🧠 Methodology Overview

1. **Interactive Visual Analytics**  
   - Folium map for launch sites and success distribution  
   - Plotly charts: payload vs. success, site‑level performance

2. **Predictive Modeling**  
   - Models: SVM, Decision Tree, Logistic Regression, KNN  
   - Hyperparameter tuning via GridSearchCV  
   - Evaluation: accuracy, confusion matrix

3. **Final Report & Insights**  
   - Business framing, executive summary, recommendations

---

## ✅ Executive Summary

This project analyzes SpaceX Falcon 9 launch records and develops a predictive model for launch success. The analysis is framed as if supporting a competing company that aims to bid for rocket launches at lower cost.

**Key findings:**
- **Launch site** strongly influences success rates.  
- **Payload mass** shows a significant relationship with outcomes, with mid‑range payloads yielding higher success.  
- **Booster version and orbit type** also play a role in predicting reliability.

By applying machine learning models such as Decision Trees, Logistic Regression, KNN, and SVM, the project achieved reasonably accurate predictions of launch success. These results provide actionable insights for decision‑making when planning or bidding for future launches.

This notebook demonstrates the complete data science workflow: data collection, exploratory analysis, interactive visualization, machine learning, and final business insights.

---

## 📈 Results & Insights

After hyperparameter tuning, Decision Trees and SVM delivered the strongest results, while Logistic Regression and KNN performed slightly lower.

**Interpretation of results:**
- **Payload Mass:** Launches in the 2,000–8,000 kg range are more likely to succeed.  
- **Launch Site:** Certain sites consistently outperform others in reliability.  
- **Booster Version:** Newer boosters demonstrate improved success rates.

The best model balances accuracy with interpretability, making it a valuable tool for forecasting future launches and supporting strategic decisions.

---

## 🧾 Conclusion

This project illustrates the end‑to‑end data science lifecycle, including:
- Collecting and wrangling launch data  
- Performing exploratory and visual analysis  
- Building interactive dashboards  
- Training and evaluating machine learning models

**Key insights:**
- Launch success depends on site, booster type, and payload.  
- The chosen model predicts outcomes with reasonable accuracy.

This work highlights how data‑driven methods can support stakeholders in planning, investing in, or competing for rocket launches.

---

## 🛠️ Setup & Usage

1) **Clone and create an environment**
```bash
git clone https://github.com/AdnanAbdelkarim/IBM-Data-Science-Professional-Certificate.git
cd applied-data-science-capstone

python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows
# .venv\Scripts\activate
```

2) **Install dependencies**
```bash
pip install -r requirements.txt
```

3) **Launch the notebook**
```bash
jupyter lab  # or: jupyter notebook
```
Open: `notebooks/IBM_Capstone_Final_Report.ipynb`

### (Optional) Run the Dash demo
```bash
python scripts/spacex_dash_app.py
```
Then open the local URL printed in the console.

---

## 📦 Requirements

```
pandas
numpy
scikit-learn
plotly
dash
folium
jupyter
```

> Keep data files small in Git; large raw files should be linked or released separately.

---

## 📜 License & Attribution

- This project was completed as part of the **IBM Data Science Professional Certificate**.  
- Code authored by **Adnan Abdelkarim**.

---

## 🙌 Acknowledgements

Thanks to IBM & Coursera instructors for the dataset and guidance, and to peers/reviewers for feedback during the capstone.
