# Exercise_1 
# Use the .head() method to view the contents of matrices a and b
print("Matrix A:")
print (a.head())

print("Matrix B:")
print (b.head())

# Complete the matrix with the product of matrices a and b
product = np.array([[10,12], [15,18]])

# Run this validation to see how your estimate performs
product == np.dot(a,b)

--------------------------------------------------
# Exercise_2 
# Print the dimensions of C
print(C.shape)

# Print the dimensions of D
print(D.shape)

# Can C and D be multiplied together?
C_times_D = None

--------------------------------------------------
# Exercise_3 
# Take a look at Matrix G using the following print function
print("Matrix G:")
print(G)

# Take a look at the matrices H, I, and J and determine which pair of those matrices will produce G when multiplied together. 
print("Matrix H:")
print(H)
print("Matrix I:")
print(I)
print("Matrix J:")
print(J)

# Multiply the two matrices that are factors of the matrix G
prod = np.matmul(H,J)
print(G == prod)

--------------------------------------------------
# Exercise_4 
# View the L, U, W, and H matrices.
print("Matrices L and U:") 
print(L)
print(U)

print("Matrices W and H:")
print(W)
print(H)

# Calculate RMSE for LU
print("RMSE of LU: ", getRMSE(LU, M))

# Calculate RMSE for WH
print("RMSE of WH: ", getRMSE(WH, M))

--------------------------------------------------
# Exercise_5 
#1
# View left factor matrix
print(U)
#2
# View right factor matrix
print(P)
#3
# Multiply factor matrices
UP = np.matmul(U,P)

# Convert to pandas DataFrame
print(pd.DataFrame(UP, columns = P.columns, index = U.index))


--------------------------------------------------
# Exercise_6 
# Use getRMSE(preds, actuals) to calculate the RMSE of matrices T and F1.
getRMSE(F1, T)

# Create list of F2, F3, F4, F5, and F6
Fs = [F2, F3, F4, F5, F6]

# Calculate RMSEs for F2 - F6
getRMSEs(Fs, T)

--------------------------------------------------
# Exercise_7 
# Import monotonically_increasing_id and show R
from pyspark.sql.functions import monotonically_increasing_id
R.show()

# Use the to_long() function to convert the dataframe to the "long" format.
ratings = to_long(R)
ratings.show()

# Get unique users and repartition to 1 partition
users = ratings.select("User").distinct().coalesce(1)

# Create a new column of unique integers called "userId" in the users dataframe.
users = users.withColumn("userId", monotonically_increasing_id()).persist()
users.show()

--------------------------------------------------
# Exercise_8 
# Extract the distinct movie id's
movies = ratings.select("Movie").distinct() 

# Repartition the data to have only one partition.
movies = movies.coalesce(1) 

# Create a new column of movieId integers. 
movies = movies.withColumn("movieId", monotonically_increasing_id()).persist() 

# Join the ratings, users and movies dataframes
movie_ratings = ratings.join(users, "User", "left").join(movies, "Movie", "left")
movie_ratings.show()

--------------------------------------------------
# Exercise_9 
# Split the ratings dataframe into training and test data
(training_data, test_data) = ratings.randomSplit([0.8, 0.2], seed=42)

# Set the ALS hyperparameters
from pyspark.ml.recommendation import ALS
als = ALS(userCol="userId", itemCol="movieId", ratingCol="rating", rank = 10, maxIter = 15, regParam = .1,
          coldStartStrategy="drop", nonnegative = True, implicitPrefs = False)

# Fit the mdoel to the training_data
model = als.fit(training_data)

# Generate predictions on the test_data
test_predictions = model.transform(test_data)
test_predictions.show()

--------------------------------------------------
# Exercise_10 
# Import RegressionEvaluator
from pyspark.ml.evaluation import RegressionEvaluator

# Complete the evaluator code
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")

# Extract the 3 parameters
print(evaluator.getMetricName())
print(evaluator.getLabelCol())
print(evaluator.getPredictionCol())

--------------------------------------------------
# Exercise_11 
# Evaluate the "predictions" dataframe
RMSE = evaluator.evaluate(test_predictions)

# Print the RMSE
print (RMSE)

--------------------------------------------------
