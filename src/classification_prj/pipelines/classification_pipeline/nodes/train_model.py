from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


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
