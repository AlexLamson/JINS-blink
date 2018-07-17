from sklearn.ensemble import RandomForestClassifier

def get_model():
	model = RandomForestClassifier(n_estimators=20,min_samples_split=4,min_samples_leaf=2, n_jobs=-1,verbose=True)
	return model


def tune_hyperparameters():
	print("test")
	pass
