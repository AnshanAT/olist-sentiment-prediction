# olist-sentiment-prediction
Repository for COMP90051 Project, Group 55


# Run this for requirements...
```
pip install -r requirements.txt
```

# Research Queston
To what extent can customer review sentiment be inferred from structured behavioural and transactional signals—such as order timing, payment behaviour, freight ratios, and product diversity—without using any textual review information?

# Dataset
We plan to use the Olist Brazilian E-commerce dataset, which consists of multiple relational tables (orders, payments, products, customers, and reviews).
https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce/data 

# Feature Engineering
To address the research question, we propose to design structured behavioural features that capture latent customer states, rather than relying on raw variables.
1.	A frustration index, defined as the normalized difference between estimated and actual delivery time, along with freight-to-price ratios
2.	A financial burden index, based on the relationship between total payment value and the number of installments
3.	A consistency index, capturing variabili    ty in customer purchasing behaviour, such as product category diversity and price dispersion

# Experimental Design
We will compare three models with different levels of complexity
1.	Linear Regression as a simple baseline
2.	Random Forest as a medium-complexity model
3.	Neural Oblivious Decision Ensembles (NODE, ICLR 2020) as a complex model

We will evaluate whether motif-based engineered features improve prediction performance compared to raw transactional features, and whether complex models such as NODE better capture non-linear interactions between these behavioural motifs.
