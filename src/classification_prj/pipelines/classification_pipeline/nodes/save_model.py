
def save_model(rf_model,path):
    rf_model.write().overwrite().parquet(path)
