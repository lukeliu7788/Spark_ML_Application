# Import all necessary libraries
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

import numpy as np
import pandas as pd

spark = SparkSession     .builder     .appName("NaiveBayesExample")     .getOrCreate()

# Get the train set path
train_datafile = "hdfs://soit-hdp-pro-1.ucc.usyd.edu.au/share/MNIST/Train-label-28x28.csv"

# Get the test set path
test_datafile = "hdfs://soit-hdp-pro-1.ucc.usyd.edu.au/share/MNIST/Test-label-28x28.csv"

# Load the train set data in csv file
train_df = spark.read.csv(train_datafile, header=False, inferSchema="true")

# convert train set to dataframe with schema["label", "features"]
train_assembler = VectorAssembler(inputCols=train_df.columns[1:], outputCol="features")
train = train_assembler.transform(train_df).selectExpr("_c0 as label","features")

# Create the trainer and set its parameters
nb = NaiveBayes(smoothing=1.0, modelType="multinomial")

# Train the model
model = nb.fit(train)

# Load the test set data in csv file
test_df = spark.read.csv(test_datafile, header=False, inferSchema="true")

#  convert test set to dataframe with schema["label", "features"]
test_assembler = VectorAssembler(inputCols=test_df.columns[1:], outputCol="features")
test = test_assembler.transform(test_df).selectExpr("_c0 as label","features")

# use the model to predict test label
predictions = model.transform(test)

# Compute accuracy on the test set
accuracyevaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = accuracyevaluator.evaluate(predictions)
print("Test set accuracy = " + str(accuracy))

# generate confusion matrix
predictions = predictions.withColumn("prediction", predictions["prediction"].cast(IntegerType()))
confusionlist=np.zeros([10, 10], np.int16)
grouplist = predictions.select("label", "prediction").collect()
for item in grouplist:
    confusionlist[item[0]][item[1]]+=1

# compute prediction, recal and  F1 value for each label
table = np.zeros([10, 4], np.float64)
for i in range(0,10):
    table[i, 0] = i
    table[i, 1] = round(float(confusionlist[i][i]*100/confusionlist.sum(axis=0)[i])/float(100), 2)
    table[i, 2] = round(float(confusionlist[i][i]*100/confusionlist.sum(axis=1)[i])/float(100), 2)
    table[i, 3] = round(float(2*(table[i, 1]*table[i, 2])/ (table[i, 1]+table[i, 2])), 2)
table = pd.DataFrame(table, index=table[:,0], columns=['label', 'prediction','recall','F1'])
print(table)

