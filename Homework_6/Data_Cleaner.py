import pandas as pd

# https://www.kaggle.com/datasets/udaymalviya/bank-loan-data
df = pd.read_csv("loan_data.csv")

quantitative_locations = [0, 3, 4, 6, 8, 9, 10, 11, 13]
feature_labels = ["person_age", "person_income", "person_emp_exp", "loan_amnt", "loan_int_rate", "loan_percent_income",
                  "cb_person_cred_hist_length", "credit_score", "loan_status"]

cleaned_df = pd.DataFrame()
MAX_ENTRIES = 2000

for i in range(len(quantitative_locations)):
    col = df.iloc[:MAX_ENTRIES, quantitative_locations[i]]
    cleaned_df.insert(i, feature_labels[i], col, True)

cleaned_df.to_csv("cleaned_data.csv")

cleaned_all_df = pd.DataFrame()
for i in range(len(quantitative_locations)):
    col = df.iloc[:, quantitative_locations[i]]
    cleaned_all_df.insert(i, feature_labels[i], col, True)

cleaned_all_df.to_csv("all_cleaned_data.csv")
