"""
This is a boilerplate pipeline 'classification_pipeline'
generated using Kedro 0.18.14
"""

from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import SparkSession


def init_spark():
    spark = SparkSession.builder.appName("PySparkClassification").getOrCreate()
    return spark


def load_data(spark, internal_data_file_path,external_sources_file_path):
    df_data = spark.read.csv(internal_data_file_path, header=True, inferSchema=True)
    df_ext = spark.read.csv(external_sources_file_path, header=True, inferSchema=True)
    df_full = df_data.join(df_ext, on='SK_ID_CURR', how='inner')
    return df_full


def preprocess_data(df_full):
    categorical_columns = ['NAME_EDUCATION_TYPE', 'CODE_GENDER', 'ORGANIZATION_TYPE', 'NAME_INCOME_TYPE']
    indexers = [StringIndexer(inputCol=col, outputCol=col + "_Index", handleInvalid="skip") for col in
                categorical_columns]

    df_indexed = df_full
    for indexer in indexers:
        df_indexed = indexer.fit(df_indexed).transform(df_indexed)

    columns_extract = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH', 'DAYS_EMPLOYED'] + \
                      [col + "_Index" for col in categorical_columns] + \
                      ['AMT_ANNUITY', 'DAYS_REGISTRATION', 'AMT_GOODS_PRICE', 'AMT_CREDIT',
                       'DAYS_LAST_PHONE_CHANGE', 'AMT_INCOME_TOTAL', 'OWN_CAR_AGE', 'TARGET']

    df = df_indexed.select(columns_extract)

    df = df.withColumnRenamed("NAME_EDUCATION_TYPE_Index", "NAME_EDUCATION_TYPE")
    df = df.withColumnRenamed("CODE_GENDER_Index", "CODE_GENDER")
    df = df.withColumnRenamed("ORGANIZATION_TYPE_Index", "ORGANIZATION_TYPE")
    df = df.withColumnRenamed("NAME_INCOME_TYPE_Index", "NAME_INCOME_TYPE")

    df_cleaned = df.na.drop()

    return df_cleaned


def train_model(df_cleaned):
    train, test = df_cleaned.randomSplit([0.8, 0.2], seed=101)
    feature_columns = [col for col in train.columns if col != 'TARGET']
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    train = assembler.transform(train)
    test = assembler.transform(test)
    scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
    scaler_model = scaler.fit(train)
    train = scaler_model.transform(train)
    test = scaler_model.transform(test)
    rf_classifier = RandomForestClassifier(featuresCol="scaled_features", labelCol="TARGET", numTrees=100, seed=50)
    rf_model = rf_classifier.fit(train)
    predictions = rf_model.transform(test)
    evaluator = MulticlassClassificationEvaluator(labelCol="TARGET", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)

    return rf_model, accuracy


def save_model(rf_model, path):
    rf_model.write().overwrite().parquet(path)
