from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructType, StructField, IntegerType
from pyspark.sql.functions import rand, col, monotonically_increasing_id

from pipelines_config import *


spark = SparkSession.builder.appName("spark").getOrCreate()

docword_schema = StructType(
    [
        StructField("doc_id", IntegerType(), True),
        StructField("word_id", IntegerType(), True),
        StructField("count", IntegerType(), True),
    ]
)


def read_docword(docword_path: str) -> DataFrame:
    """Reads dataframe from docword_path and removes first three rows

    Args:
        docword_path (str)

    Returns:
        DataFrame
    """
    logger.info(f"Started reading docword from {docword_path}")
    df = (
        spark.read.format("csv")
        .option("delimiter", True)
        .option("header", False)
        .option("delimiter", " ")
        .schema(docword_schema)
        .load(docword_path)
    )

    logger.info(f"Started deleting first 3 rows from {docword_path}")
    df = df.withColumn("id", monotonically_increasing_id())
    df = df.where(df.id > 2)
    df = df.drop("id")

    return df


def sample_docword(docword_df: DataFrame, sample_size: int) -> DataFrame:
    """From docword_df samples sample_size documents i.e. returns all
    entries corresponding to randomly chosen documents.

    Args:
        docword_df (DataFrame)
        sample_size (int)

    Returns:
        DataFrame
    """
    logger.info(f"Started computing documents count")
    docs_count = docword_df.select("doc_id").distinct().count()
    logger.info(f"Started calculating sample size")
    real_sample_size = min(sample_size, docs_count)
    logger.info(f"Final sample size is {real_sample_size}")

    logger.info(f"Started sampling dataframe")
    docs_sample = (
        docword_df.select("doc_id")
        .distinct()
        .sample(fraction=real_sample_size / docs_count)
    )
    return docword_df.join(docs_sample, "doc_id", "inner")


def main():
    """
    From each docword.{name}.txt.gz file gets sample_size documents and
    saves to docword.{name}.sample.csv files. Sample size is specified
    in pipelines_config.py. If sample size if bigger than document count
    in DataFrame then entire DataFrame is returned
    """
    logger.info("STARTED SAMPLING PIPELINE")
    for dataset_name, docword_path, docword_sampled_path in zip(
        dataset_names, docword_paths, docwords_sampled_paths
    ):
        logger.info(f"Started processing {dataset_name}")
        df = read_docword(docword_path)
        sample = sample_docword(df, sample_size=sample_size)
        logger.info(f"Started converting sample to Pandas dataframe")
        sample = sample.toPandas()
        logger.info("Displaying head of output dataframe")
        print(sample.head(5))
        logger.info(f"Started saving sample into {docword_sampled_path}")
        sample.to_csv(docword_sampled_path, index=False)


if __name__ == "__main__":
    main()
