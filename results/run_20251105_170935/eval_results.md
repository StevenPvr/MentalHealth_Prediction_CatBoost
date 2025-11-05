# Evaluation Results

- Timestamp: 20251105_170935

## Global Metrics

- Logloss: 0.4764
- AUC: 0.8382
- Accuracy: 0.7519
- F1: 0.7613
- Recall: 0.8056

## Fairness by Gender

- Overall AUC: 0.8382
- AUC gap: 0.0543
  - female: AUC=0.8796 | logloss=0.3683 | n=6234
  - male: AUC=0.8253 | logloss=0.4911 | n=46032

## Fairness by Country

- Overall AUC: 0.8382
- AUC gap: 0.0991
  - canada: AUC=0.8979 | logloss=0.3731 | n=2815
  - others: AUC=0.8794 | logloss=0.3847 | n=8979
  - uk: AUC=0.8500 | logloss=0.4368 | n=9522
  - usa: AUC=0.7988 | logloss=0.5247 | n=30950

## SHAP

![SHAP Summary](/Users/steven/Documents/Programmation/MentalHealth_CatBoost/results/plots/shap_summary.png)

## SHAP vs Cramér's V

| Feature | Mean SHAP | Cramér's V |
| --- | ---: | ---: |
| familyhistory | 0.0991 | 0.3636 |
| mentalhealthinterview | -0.0566 | 0.0905 |
| country | -0.0535 | 0.1244 |
| careoptions | -0.0216 | 0.2967 |
| gender | 0.0186 | 0.1467 |
| workinterest | -0.0056 | 0.0045 |
| occupation | 0.0043 | 0.0102 |
| selfemployed | 0.0040 | 0.0370 |
| socialweakness | -0.0037 | 0.0032 |
| habitschange | -0.0020 | 0.0035 |
| moodswings | 0.0019 | 0.0048 |
| mentalhealthhistory | -0.0007 | 0.0104 |
| daysindoors | 0.0005 | 0.0078 |
| increasingstress | 0.0000 | 0.0100 |
| copingstruggles | -0.0000 | 0.0071 |
