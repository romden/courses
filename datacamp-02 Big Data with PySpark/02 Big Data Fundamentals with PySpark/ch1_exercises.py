# Exercise_1 
# Print the version of SparkContext
print("The version of Spark Context in the PySpark shell is", sc.version)

# Print the Python version of SparkContext
print("The Python version of Spark Context in the PySpark shell is", sc.pythonVer)

# Print the master of SparkContext
print("The master of Spark Context in the PySpark shell is", sc.master)

--------------------------------------------------
# Exercise_2 
# Create a python list of numbers from 1 to 100  
numb = range(1, 101)

# Load the list into PySpark
spark_data = sc.parallelize(numb)

--------------------------------------------------
# Exercise_3 
# Load a local file into PySpark shell
lines = sc.textFile(file_path)

--------------------------------------------------
# Exercise_4 
# Print my_list in the console
print("Input list is", my_list)

# Square all numbers in my_list 
squared_list_lambda = list(map(lambda x: x**2, my_list))

# Print the result of the map function
print("The squared numbers are", squared_list_lambda)

--------------------------------------------------
# Exercise_5 
# Print my_list2 in the console
print("Input list is:", my_list2)

# Filter numbers divisible by 10
filtered_list = list(filter(lambda x: (x%10 == 0), my_list2))

# Print the numbers divisible by 10
print("Numbers divisible by 10 are:", filtered_list)

--------------------------------------------------
