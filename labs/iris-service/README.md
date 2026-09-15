# מעבדה: מודל שהופך לשירות

הקוד שמלווה את פרקים 18–24 במדריך. כל קובץ כאן רץ ונבדק.

## הפעלה

```bash
python -m venv .venv
source .venv/bin/activate          # ב-Windows: .venv\Scripts\activate.bat
python -m pip install -r requirements-dev.txt

python train.py                    # יוצר את artifacts/
python -m pytest -q                # 7 בדיקות
python -m uvicorn app:app --port 8000
```

אחר כך פתח `http://127.0.0.1:8000/docs` ושלח בקשה ל-`/predict`.

## קבצים

| קובץ | תפקיד |
| --- | --- |
| `train.py` | אימון, Baseline, חיפוש הגדרות והערכה אחת על סט הבדיקה |
| `app.py` | שירות חיזוי עם חוזה קלט, בדיקות בריאות ולוג מובנה |
| `test_app.py` | בדיקות חוזה, סדר תכונות ובדיקת מודל מול Baseline |
| `track_run.py` | רישום הריצה ב-MLflow |
| `Dockerfile` | אריזת השירות לקונטיינר שרץ כמשתמש לא-root |
| `github-actions-ci.yml` | דוגמת Pipeline: בדיקות, שער איכות, בנייה ובדיקת עשן |
| `Makefile` | קיצורי דרך: `make train`, `make test`, `make serve` |

## תרגילי תקלה

1. החלף בין `petal_length_cm` ל-`sepal_length_cm` ברשימת `FEATURES` והרץ מחדש. איזו בדיקה נכשלת?
2. מחק את `artifacts/model.joblib` והפעל את השירות. מה קורה, ומתי זה מתגלה?
3. שנה את `test_macro_f1` בשער האיכות ל-0.99 ובדוק שה-Pipeline עוצר.
4. הוסף שדה חדש לבקשה ובדוק שהוא נדחה עם 422.
