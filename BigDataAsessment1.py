from pyspark.sql import SparkSession
from pyspark.sql.functions import isnull, when, count,col
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline
import matplotlib.pyplot as plt
import numpy as np

#Create Spark Session
spark = SparkSession.builder.appName("Assessment").getOrCreate()
sc = spark.sparkContext

#Pre-Processing
df1 = spark.read.csv("Car_Prices.csv",header=True,inferSchema=True)
print("Dataset")
df1.show(10)#Show the dataset
print("Check for nulls")
df1.select([count(when(col(c).isNull(), c)).alias(c) for c in df1.columns]).show()#Checks for nulls
print("Schema")
df1.printSchema()#View the schema of the dataset

#Drops null data if present
#df1 = df1.dropna()
cols_to_index = ["fuel","seller_type","transmission","owner"]#List of columns we wish to index
indexer = [StringIndexer(inputCol=column, outputCol=column+"_index").fit(df1) for column in list(cols_to_index)]#Creates and indexer and fits it to the dataset
pipeline = Pipeline(stages=indexer)#Create a pipeline to transform my indexs
df_r = pipeline.fit(df1)#Fit the dataframe to the pipeline
Processed_df=df_r.transform(df1)#Transforms data with indexer and outputs to new dataframe
print("Processed data")
Processed_df.show(10)

#Make features - only those wished to be used for prediction of car price
assembler=VectorAssembler(inputCols=['year','km_driven','fuel_index','seller_type_index','transmission_index','owner_index'],outputCol='features')#Creates a feature column with the desired input columns
output=assembler.transform(Processed_df)#Transformers it with processed data
output.show(10)
#Complie final data with out features and label
final_data = output.select("features", "selling_price")

#Test train slipt of 70% train, 30% test
train_data,test_data = final_data.randomSplit([0.7,0.3])

#Create linear regression
model = LinearRegression(featuresCol='features',labelCol='selling_price', maxIter=10, regParam=0.3, elasticNetParam=0.8).fit(train_data)
summary = model.evaluate(test_data)
summarytrain = model.evaluate(train_data)

#Modle evlaution
print("Coefficients: " + str(model.coefficients))#Slope of line
print("Intercept: " + str(model.intercept))#Entry point of line

#Train evaluation with MAE MSE RMSE
print("Train Data:")
print("MAE: " + str(summarytrain.meanAbsoluteError))
print("RMSE: " + str(summarytrain.rootMeanSquaredError))
print("MSE: " + str(summarytrain.meanSquaredError))

#Test evaluation with MAE MSE RMSE
print("Test Data:")
print("MAE: " + str(summary.meanAbsoluteError))
print("RMSE: " + str(summary.rootMeanSquaredError))
print("MSE: " + str(summary.meanSquaredError))

#Predictions
predictions = model.transform(test_data)
predictions.show(10)

#Convert to numpy arrays for plotting - get around a recursion error
price_plot = predictions.select("selling_price").toPandas().to_numpy()
pridiction_plot = predictions.select("prediction").toPandas().to_numpy()

#Flatten to a 1d array for line of best fit
price_plot = price_plot.ravel()
pridiction_plot = pridiction_plot.ravel()

#Scatter plot for compoarion of actual price v predicted price
plt.scatter(pridiction_plot, price_plot)

#Plot line of best fit
theta = np.polyfit(pridiction_plot, price_plot, 1)
y_line = theta[1] + theta[0] * pridiction_plot
plt.plot(pridiction_plot, y_line, 'r')
plt.title("Predicted vs Actual Price in hundradthousands")
plt.xlabel("Predicted Price")
plt.ylabel("Actual Price")
plt.show()