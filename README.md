# novartis-datathon
This repository is going to be used to solve the NOVARTIS 2024 Datathon Challenge


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




