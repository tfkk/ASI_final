import h2o
from h2o.automl import H2OAutoML

h2o.init()

data = h2o.import_file("../data/ds_salaries.csv")

aml = H2OAutoML(max_runtime_secs=3600, seed=42)

aml.train(y="salary", training_frame=data)

leaderboard = aml.leaderboard
print(leaderboard)


best_model = aml.leader

best_model.save_mojo("./model")
