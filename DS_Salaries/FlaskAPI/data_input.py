import json
import os

import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(__file__)
DEFAULT_DATA_PATH = os.path.join(BASE_DIR, "..", "salary_data_cleanedv2.csv")


def get_x_test(data_path=DEFAULT_DATA_PATH):
	df = pd.read_csv(data_path)
	df_model = df[
		[
			"Rating",
			"State",
			"Same State",
			"Size",
			"Type of ownership",
			"Industry",
			"Sector",
			"Revenue",
			"Competitor Count",
			"Hourly",
			"Employer Provided",
			"Min Salary",
			"Max Salary",
			"Avg Salary",
			"Company Age",
			"Python",
			"R",
			"AWS",
			"Spark",
			"Excel",
			"Job Simplified",
			"Seniority",
			"Desc Length",
		]
	]
	df_dum = pd.get_dummies(df_model)
	X = df_dum.drop("Avg Salary", axis=1)
	y = df_dum["Avg Salary"].values
	_, X_test, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)
	return X_test


def get_sample_payload(index=1, data_path=DEFAULT_DATA_PATH):
	X_test = get_x_test(data_path=data_path)
	return {"features": X_test.iloc[index, :].to_dict()}


if __name__ == "__main__":
	sample_payload = get_sample_payload(index=1)
	print(json.dumps(sample_payload, indent=2))

