# Sales Forecasting – Gradient-Boosted Trees + GridSearch

Files
- train.py        : cleaning, features, GridSearchCV, evaluation; writes model.joblib, meta.json, bundle.json
- app.py          : Flask REST API (POST /predict, GET /health, GET /metrics)
- template.html   : web app source (HTML/CSS/JS)
- build.py        : embeds bundle.json into template.html -> sales_forecast_app.html
- sales_forecast_app.html : final standalone app (just double-click to open)

Run
1. pip install pandas scikit-learn joblib flask
2. Put the CSV next to train.py and edit the path at the top of train.py
3. python train.py        # trains + exports model
4. python build.py        # builds the web app
5. python app.py          # optional: REST API on http://localhost:5000
