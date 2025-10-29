# Evaluation Results

- Timestamp: 20251028_232636

## Global Metrics

- Logloss: 0.4802
- AUC: 0.8381
- Accuracy: 0.7518
- F1: 0.7598
- Recall: 0.7991

## Fairness by Gender

- Overall AUC: 0.8381
- AUC gap: 0.0535
  - female: AUC=0.8790 | logloss=0.3778 | n=6234
  - male: AUC=0.8254 | logloss=0.4941 | n=46032

## Fairness by Country

- Overall AUC: 0.8381
- AUC gap: 0.1008
  - canada: AUC=0.9003 | logloss=0.3795 | n=2815
  - others: AUC=0.8791 | logloss=0.3906 | n=8979
  - uk: AUC=0.8496 | logloss=0.4457 | n=9522
  - usa: AUC=0.7995 | logloss=0.5260 | n=30950

## SHAP

![SHAP Summary](/Users/steven/Documents/Programmation/MentalHealth/plots/shap_summary.png)

## SHAP vs Cramér's V

| Feature | Mean SHAP | Cramér's V |
| --- | ---: | ---: |
| familyhistory | -0.0201 | 0.3611 |
| selfemployed | -0.0170 | 0.0380 |
| gender | 0.0164 | 0.1473 |
| increasingstress | 0.0106 | 0.0053 |
| mentalhealthinterview | -0.0088 | 0.0909 |
| country | 0.0033 | 0.1230 |
| workinterest | 0.0019 | 0.0033 |
| occupation | 0.0017 | 0.0091 |
| daysindoors | -0.0015 | 0.0041 |
| habitschange | 0.0013 | 0.0010 |
| socialweakness | -0.0010 | 0.0032 |
| mentalhealthhistory | 0.0007 | 0.0053 |
| moodswings | 0.0006 | 0.0036 |
| careoptions | -0.0005 | 0.2897 |
| copingstruggles | 0.0000 | 0.0067 |
