import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Load the EC2 dataset
data = pd.read_csv("ec2dataset.csv")

# Display basic information about the dataset
print(data.info())

# Preview the first 5 rows
print(data.head())

# List of cost-related columns
cost_columns = [
    'On Demand',
    'Linux Reserved cost',
    'Linux Spot Minimum cost',
    'Windows On Demand cost',
    'Windows Reserved cost'
]

# Remove '$', commas, and 'hourly', then convert to numeric
for column in cost_columns:
    data[column] = pd.to_numeric(
        data[column].str.replace('[$, hourly]', '', regex=True),
        errors='coerce'
    )

# Check for missing values after conversion
print("\nMissing values after cleaning:")
print(data[cost_columns].isnull().sum())

# Generate summary statistics for cost-related columns
cost_summary = data[cost_columns].describe()

print("\nCost Summary Statistics:")
print(cost_summary)



# Set up the plotting style
sns.set(style="whitegrid")

# Create a boxplot to visualize the distribution of costs
plt.figure(figsize=(12, 6))

sns.boxplot(
    data=data[cost_columns],
    palette="Set2"
)

# Add title and labels
plt.title("Cost Comparison of Amazon EC2 Instances (Hourly)", fontsize=16)
plt.ylabel("Cost (USD)", fontsize=12)
plt.xticks(rotation=45, ha="right", fontsize=12)

# Make everything fit nicely
plt.tight_layout()

# Display the graph
#plt.show()

# Function to identify outliers using the IQR method
def detect_outliers(column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    return data[
        (data[column] < lower_bound) |
        (data[column] > upper_bound)
    ]


# Find outliers in the On-Demand cost column
outliers_on_demand = detect_outliers('On Demand')

print("\nOn-Demand Cost Outliers:")
print(outliers_on_demand)

# Compare Reserved and On-Demand costs
cost_comparison = data[
    ['Name', 'On Demand', 'Linux Reserved cost']
].dropna().sort_values('On Demand')

print("\nTop 10 Cost-Effective EC2 Instances:")
print(cost_comparison.head(10))

#Reserved instances provide significant cost savings compared to On-Demand instances.
#In the analyzed data, the lowest-cost instances show savings of approximately 37–38% 
#when using Linux Reserved pricing instead of On-Demand pricing. Therefore, Reserved instances
#can be more cost-effective for workloads that are expected to run consistently.

# Step 8: Filter and Compare Different Instance Families

# Function to filter a specific instance family
def filter_instance_family(family):
    return data[data['Name'].str.startswith(family)]

# Filter T2 and T3 instance families
t2_instances = filter_instance_family('T2')
t3_instances = filter_instance_family('T3')

# Generate summary statistics
t2_summary = t2_instances[cost_columns].describe()
t3_summary = t3_instances[cost_columns].describe()

print("\nT2 Instance Costs Summary:")
print(t2_summary)

print("\nT3 Instance Costs Summary:")
print(t3_summary)

# Visualize cost distribution for T2 instances
plt.figure(figsize=(12, 6))

sns.boxplot(
    data=t2_instances[cost_columns],
    palette="Blues",
    showmeans=True
)

plt.title("Cost Distribution for T2 Instances", fontsize=16)
plt.ylabel("Cost (USD)", fontsize=12)
plt.xticks(rotation=45, ha="right", fontsize=12)
plt.tight_layout()
#plt.show()


# Visualize cost distribution for T3 instances
plt.figure(figsize=(12, 6))

sns.boxplot(
    data=t3_instances[cost_columns],
    palette="Greens",
    showmeans=True
)

plt.title("Cost Distribution for T3 Instances", fontsize=16)
plt.ylabel("Cost (USD)", fontsize=12)
plt.xticks(rotation=45, ha="right", fontsize=12)
plt.tight_layout()
#plt.show()

# Compare On-Demand and Reserved costs for T2 and T3 families
comparison = pd.concat([
    t2_instances[['Name', 'On Demand', 'Linux Reserved cost']],
    t3_instances[['Name', 'On Demand', 'Linux Reserved cost']]
])

# Sort by On-Demand cost
comparison_sorted = comparison.dropna().sort_values('On Demand')

print("\n10 Lowest-Cost T2 and T3 Instances:")
print(comparison_sorted.head(10))

# -----------------------------
# PART 2: Regression Analysis
# -----------------------------

# Step 3: Feature Engineering

# Convert Instance Memory from text such as "0.5 GiB" to a number
data['Instance Memory'] = pd.to_numeric(
    data['Instance Memory'].str.replace(' GiB', '')
)

# Extract the number from vCPUs, for example "2 vCPUs" becomes 2
data['vCPUs'] = pd.to_numeric(
    data['vCPUs'].str.extract(r'(\d+)', expand=False)
)

# Check the converted data
print("\nMemory and vCPU after conversion:")
print(data[['Instance Memory', 'vCPUs']].head())

# Step 4: Handle Missing Data

# Remove rows with missing values in the columns needed for regression
data_cleaned = data.dropna(
    subset=['On Demand', 'Instance Memory', 'vCPUs']
)

# Check for missing values
print("\nMissing values after preparing regression data:")
print(
    data_cleaned[
        ['On Demand', 'Instance Memory', 'vCPUs']
    ].isnull().sum()
)

# Show how many rows remain
print("\nNumber of rows available for regression:", len(data_cleaned))

# Step 5: Split the Data

from sklearn.model_selection import train_test_split

# Features used to predict the price
X = data_cleaned[['Instance Memory', 'vCPUs']]

# Target value we want to predict
y = data_cleaned['On Demand']

# Split into 80% training data and 20% testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# Step 6: Train a Linear Regression Model

from sklearn.linear_model import LinearRegression

# Create the linear regression model
model = LinearRegression()

# Train the model using the training data
model.fit(X_train, y_train)

# Display the model's intercept and coefficients
print("\nLinear Regression Model:")
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)

# Step 7: Evaluate the Model

from sklearn.metrics import mean_absolute_error, mean_squared_error

# Predict On-Demand costs for the test data
y_pred = model.predict(X_test)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5

print("\nModel Evaluation:")
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)

#The linear regression model achieved an MAE of 1.81 and an RMSE of 5.19. The higher RMSE indicates
#  that some EC2 instances have relatively large prediction errors.
#  This suggests that memory and vCPUs alone may not fully explain differences in EC2 On-Demand pricing.


# Step 8: Visualize Actual vs Predicted Costs

plt.figure(figsize=(8, 6))

# Plot actual prices against predicted prices
plt.scatter(y_test, y_pred, alpha=0.7, color='b')

# Perfect prediction reference line
plt.plot(
    [min(y_test), max(y_test)],
    [min(y_test), max(y_test)],
    color='red',
    linestyle='--'
)

plt.title("Actual vs Predicted On-Demand Costs")
plt.xlabel("Actual On-Demand Cost")
plt.ylabel("Predicted On-Demand Cost")

plt.tight_layout()
plt.show()

# Step 9: Make a Prediction

# New instance with 4 GiB memory and 2 vCPUs
new_instance = [[4, 2]]

# Predict its On-Demand cost
predicted_cost = model.predict(new_instance)

print(
    f"\nPredicted On-Demand Cost for 4 GiB, 2 vCPUs: "
    f"${predicted_cost[0]:.4f}"
)