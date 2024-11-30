# novartis-datathon
This repository is going to be used to solve the NOVARTIS 2024 Datathon Challenge

## Possibles opcions
- MODEL LGTBIQ+:
  - Imputació k-nn
  - Imputació MICE
  - Imputació Random Forest

Amb diferents mètodes per escalar (max-min, box-cox, z-score...)

**Per fer demà**: afegir la nostra mètrica als paràmetres del model.

- MODEL XGBoost
  - Imputació k-nn
  - Imputació MICE
  - Imputació Random Forest

Amb diferents mètodes per escalar i dades sense escalar.

## Millor model de moment
- MODEL LGTBIQ+:
  - Imputació k-nn
    - dades escalades standard
   
## Models descartats
- MODEL LGTBIQ+ ; Imputació k-nn ; sense escalar
- MODEL LGTBIQ+ ; Sense imputació ; escalades standard



# Data Cleaning

Creem nova variable: time difference = data (date) - data de llançament (launch_date)
Determinem NaN
Dimensions: 118917 x 19 
Drop de ind_launch_date 87797 ~ 74% (missings) i de cluster_nl (combinació de dues vars)
Convertim a format data el necessari

2 entregues divendres 29/11: 
- Escalant variables numèriques (menys target)
- Sense escalar

Imputem els valors faltants amb KNN


# Mètriques

LightGBM ; imputació K-nn ; escalades

boosting_type: gbdt
  - learning_rate = 0.05 --> 0.1628
  - learning_rate = 0.06 --> 0.1620

learning_rate = 0.09 --> 0.1613
  - num_leaves = 63 --> 0.1607
  - num_leaves = 64 --> 0.1606

num_leaves = 70 --> 0.1597
  - lambda l1 = 0.5 ; lambda_l2 = 5 --> 0.1595
  - lambda l1 = 1 ; lambda_l2 = 10 --> 0.1587 

lambda l1 = 2 ; lambda_l2 = 20 --> 0.1581






