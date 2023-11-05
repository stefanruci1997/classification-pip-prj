from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer


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
