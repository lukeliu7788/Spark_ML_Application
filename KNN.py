
# coding: utf-8

import findspark
findspark.init()
from pyspark.sql import SparkSession
from pyspark.ml.feature import PCA
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
import numpy as np
import matplotlib.pyplot as plt
import sys


# select the dimension an nearest neighbour number KNNs
dimension=int(sys.argv[1])
KNNs=int(sys.argv[2])
Resultname="Assignment2_D_"+str(dimension)+"_K_"+str(KNNs)+"_e_9_c_2"
spark = SparkSession     .builder     .appName(Resultname)     .getOrCreate()
# read the trainning data
train_datafile = "hdfs://soit-hdp-pro-1.ucc.usyd.edu.au/share/MNIST/Train-label-28x28.csv"
train_label_df = spark.read.csv(train_datafile,header=False,inferSchema="true")
train_assembler = VectorAssembler(inputCols=train_label_df.columns[1:],
    outputCol="features")
train_vectors = train_assembler.transform(train_label_df).selectExpr("_c0 as label","features")


# use trainning data to fit the pca model and transform the trainning data to (label,pca)
pca = PCA(k=dimension, inputCol="features", outputCol="pca")
model = pca.fit(train_vectors.select("features"))
train_pca_result = model.transform(train_vectors).select("pca")
train_labs=train_vectors.select("label")
num_train_samples=60000

# read the test data
test_datafile="hdfs://soit-hdp-pro-1.ucc.usyd.edu.au/share/MNIST/Test-label-28x28.csv"
num_test_samples = 10000
test_label_df = spark.read.csv(test_datafile,header=False,inferSchema="true")
test_assembler = VectorAssembler(inputCols=test_label_df.columns[1:],
    outputCol="features")

# use pca model before transform the test data to (testlabel,testpca)
test_vectors = test_assembler.transform(test_label_df).selectExpr("_c0 as label","features")
test_pca_result = model.transform(test_vectors).selectExpr("label as testlabel","pca as testpca")
test_pca_rdd=test_pca_result.rdd


# create a train_total numpy array which include both labels and pca features
# broadcast the numpy array
from pyspark.context import SparkContext
train_pca=np.array(train_pca_result.collect())
train_pca=train_pca.reshape((num_train_samples,dimension))
train_labels=np.array(train_labs.collect())
train_total=np.hstack((train_labels,train_pca))
sc=spark.sparkContext
train_pca_boardcast=sc.broadcast(train_total)



# use the boardcast train date to do the predict
# for each row in the text data, return (real_label,predic_label)
def KNN_predict(line):
    train_label=train_pca_boardcast.value[:,0].astype('int64')
    train_pca_temp=train_pca_boardcast.value[:,1:]
    distances=(((train_pca_temp-line[1].toArray())**2).sum(axis=1))**0.5
    resultindex=np.argsort(distances)
    resultfinal=np.argmax(np.bincount(train_label[resultindex[:KNNs]].flatten()))
    return (line[0],resultfinal)
result_predict=test_pca_rdd.map(KNN_predict)


# collect the result_predict to complete other static tasks
result_array=np.array(result_predict.collect())

# calculate a confusionlist
confusionlist=np.zeros([10,10],np.int16)
for item in result_array:
    confusionlist[item[0]][item[1]]+=1
right_number=0
for i in range(0,10):
    right_number+=confusionlist[i][i]
# use confusionlist calculate accurcy   
accurcy=float(right_number)/num_test_samples
print(accurcy)
print(confusionlist)

# use confusionlist calculate P,R,F base on the labels   
Precision=np.zeros([10])
Recall=np.zeros([10])
F1=np.zeros([10])
for k in range(0,10):
    Precision[k]=round(float(confusionlist[k][k])*100/confusionlist.sum(axis=1)[k],2)
    Recall[k]=round(float(confusionlist[k][k])*100/confusionlist.sum(axis=0)[k],2)
    F1[k]=round(2*(Recall[k]*Precision[k])/(Recall[k]+Precision[k]),2)
    
    
static_table=np.vstack((Precision,Recall,F1))


# output a PrettyTable using the data caclucated before
from prettytable import PrettyTable
labels = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)

x = PrettyTable(['Label', 'Precision', 'Recall', 'F1-score'])

for i in range(0,10):
        x.add_row([labels[i],static_table[0][i],static_table[1][i],static_table[2][i]])
        
print(x)
print("D is "+str(dimension)+", K is "+str(KNNs))


