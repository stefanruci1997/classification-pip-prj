def load_data(spark, parameters):
    df_data = spark.read.csv(parameters["internal_data_file_path"], header=True, inferSchema=True)
    df_ext = spark.read.csv(parameters["external_sources_file_path"], header=True, inferSchema=True)
    df_full = df_data.join(df_ext, on='SK_ID_CURR', how='inner')
    return df_full
