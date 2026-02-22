import mlflow

mlflow.set_tracking_uri("http://18.222.171.36:5000")
client = mlflow.tracking.MlflowClient()

models = client.search_registered_models()

for m in models:
    print("Model Name:", m.name)
    print("Aliases:", m.aliases)
    print("Latest Versions:", [(v.version, v.current_stage) for v in m.latest_versions])
    print("-----")