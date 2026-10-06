import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="EC2 Cost Analysis Dashboard",
    page_icon="☁️",
    layout="wide"
)

st.title("☁️ Amazon EC2 Cost Analysis Dashboard")

st.write(
    """
    This dashboard analyzes Amazon EC2 pricing data and uses
    Linear Regression to predict On-Demand EC2 instance costs.
    """
)


# ============================================================
# LOAD DATA
# ============================================================

data = pd.read_csv("ec2dataset.csv")

st.header("1. Dataset Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Total EC2 Instances", len(data))
col2.metric("Dataset Columns", len(data.columns))
col3.metric("Missing Values", data.isnull().sum().sum())

with st.expander("View Raw Dataset"):
    st.dataframe(data, use_container_width=True)


# ============================================================
# CLEAN COST DATA
# ============================================================

cost_columns = [
    "On Demand",
    "Linux Reserved cost",
    "Linux Spot Minimum cost",
    "Windows On Demand cost",
    "Windows Reserved cost"
]

for column in cost_columns:
    data[column] = pd.to_numeric(
        data[column].str.replace("[$, hourly]", "", regex=True),
        errors="coerce"
    )


# ============================================================
# PART 1 - EC2 COST ANALYSIS
# ============================================================

st.header("2. EC2 Cost Analysis")

st.subheader("Pricing Summary")

cost_summary = data[cost_columns].describe()

st.dataframe(
    cost_summary.round(4),
    use_container_width=True
)


# ============================================================
# OVERALL COST DISTRIBUTION
# ============================================================

st.subheader("EC2 Pricing Distribution")

fig1, ax1 = plt.subplots(figsize=(12, 6))

sns.boxplot(
    data=data[cost_columns],
    ax=ax1
)

ax1.set_title("Cost Comparison of Amazon EC2 Instances (Hourly)")
ax1.set_ylabel("Cost (USD)")
ax1.tick_params(axis="x", rotation=45)

plt.tight_layout()

st.pyplot(fig1)

st.info(
    """
    The boxplot compares different EC2 pricing models.
    Linux Spot pricing generally has lower costs, while some
    On-Demand and Windows instances contain very high-cost outliers.
    """
)


# ============================================================
# OUTLIER DETECTION
# ============================================================

st.subheader("On-Demand Cost Outliers")


def detect_outliers(df, column):

    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    return df[
        (df[column] < lower_bound) |
        (df[column] > upper_bound)
    ]


outliers = detect_outliers(data, "On Demand")

col1, col2 = st.columns(2)

col1.metric(
    "On-Demand Outliers",
    len(outliers)
)

col2.metric(
    "Maximum On-Demand Cost",
    f"${data['On Demand'].max():.2f}/hr"
)

with st.expander("View On-Demand Outliers"):

    st.dataframe(
        outliers[
            [
                "Name",
                "API Name",
                "On Demand",
                "Instance Memory",
                "vCPUs"
            ]
        ],
        use_container_width=True
    )


# ============================================================
# CHEAPEST INSTANCES
# ============================================================

st.subheader("10 Lowest-Cost EC2 Instances")

cost_comparison = data[
    [
        "Name",
        "On Demand",
        "Linux Reserved cost"
    ]
].dropna().sort_values("On Demand")

top10 = cost_comparison.head(10).copy()

top10["Reserved Savings (%)"] = (
    (
        top10["On Demand"] -
        top10["Linux Reserved cost"]
    )
    / top10["On Demand"]
    * 100
)

st.dataframe(
    top10.round(4),
    use_container_width=True
)

st.success(
    """
    Reserved pricing is lower than On-Demand pricing for these
    low-cost instances. This can provide significant savings for
    workloads that run consistently.
    """
)


# ============================================================
# T2 AND T3 FAMILY COMPARISON
# ============================================================

st.header("3. T2 vs T3 Instance Comparison")


def filter_instance_family(family):

    return data[
        data["Name"].str.startswith(family)
    ]


t2_instances = filter_instance_family("T2")
t3_instances = filter_instance_family("T3")


col1, col2 = st.columns(2)

col1.metric(
    "T2 Average On-Demand",
    f"${t2_instances['On Demand'].mean():.4f}/hr"
)

col2.metric(
    "T3 Average On-Demand",
    f"${t3_instances['On Demand'].mean():.4f}/hr"
)


# ============================================================
# T2 GRAPH
# ============================================================

col1, col2 = st.columns(2)

with col1:

    st.subheader("T2 Cost Distribution")

    fig2, ax2 = plt.subplots(figsize=(8, 6))

    sns.boxplot(
        data=t2_instances[cost_columns],
        showmeans=True,
        ax=ax2
    )

    ax2.set_ylabel("Cost (USD)")
    ax2.tick_params(axis="x", rotation=45)

    plt.tight_layout()

    st.pyplot(fig2)


# ============================================================
# T3 GRAPH
# ============================================================

with col2:

    st.subheader("T3 Cost Distribution")

    fig3, ax3 = plt.subplots(figsize=(8, 6))

    sns.boxplot(
        data=t3_instances[cost_columns],
        showmeans=True,
        ax=ax3
    )

    ax3.set_ylabel("Cost (USD)")
    ax3.tick_params(axis="x", rotation=45)

    plt.tight_layout()

    st.pyplot(fig3)


st.info(
    """
    In this dataset, T3 instances have a slightly lower average
    Linux On-Demand cost than T2 instances.
    """
)


# ============================================================
# T2 / T3 LOWEST COST COMPARISON
# ============================================================

comparison = pd.concat(
    [
        t2_instances[
            ["Name", "On Demand", "Linux Reserved cost"]
        ],
        t3_instances[
            ["Name", "On Demand", "Linux Reserved cost"]
        ]
    ]
)

comparison_sorted = (
    comparison
    .dropna()
    .sort_values("On Demand")
)

with st.expander("View 10 Lowest-Cost T2 and T3 Instances"):

    st.dataframe(
        comparison_sorted.head(10),
        use_container_width=True
    )


# ============================================================
# PART 2 - MACHINE LEARNING
# ============================================================

st.header("4. Machine Learning - EC2 Cost Prediction")

st.write(
    """
    A Linear Regression model is trained using **Instance Memory**
    and **vCPUs** to predict the Linux **On-Demand hourly cost**.
    """
)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

data["Instance Memory"] = pd.to_numeric(
    data["Instance Memory"]
    .str.replace(" GiB", ""),
    errors="coerce"
)

data["vCPUs"] = pd.to_numeric(
    data["vCPUs"]
    .str.extract(r"(\d+)", expand=False),
    errors="coerce"
)


# ============================================================
# REMOVE MISSING REGRESSION VALUES
# ============================================================

data_cleaned = data.dropna(
    subset=[
        "On Demand",
        "Instance Memory",
        "vCPUs"
    ]
)
# ============================================================
# REMOVE ON-DEMAND COST OUTLIERS FOR MACHINE LEARNING
# ============================================================

# Calculate Q1 and Q3
Q1 = data_cleaned["On Demand"].quantile(0.25)
Q3 = data_cleaned["On Demand"].quantile(0.75)

# Calculate Interquartile Range
IQR = Q3 - Q1

# Calculate acceptable price range
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Remove EC2 instances outside the normal price range
ml_data = data_cleaned[
    (data_cleaned["On Demand"] >= lower_bound) &
    (data_cleaned["On Demand"] <= upper_bound)
].copy()

st.write(
    f"Regression data before removing outliers: "
    f"**{len(data_cleaned)} instances**"
)

st.write(
    f"Regression data after removing outliers: "
    f"**{len(ml_data)} instances**"
)

st.write(
    f"Removed **{len(data_cleaned) - len(ml_data)} outliers** "
    f"before training the regression model."
)

# ============================================================
# CREATE FEATURES AND TARGET
# ============================================================

X = ml_data[
    [
        "Instance Memory",
        "vCPUs"
    ]
]

y = ml_data["On Demand"]


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ============================================================
# TRAIN LINEAR REGRESSION MODEL
# ============================================================

model = LinearRegression()

model.fit(
    X_train,
    y_train
)


# ============================================================
# MAKE TEST PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# MODEL EVALUATION
# ============================================================

mae = mean_absolute_error(
    y_test,
    y_pred
)

mse = mean_squared_error(
    y_test,
    y_pred
)

rmse = mse ** 0.5


st.subheader("Model Performance")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Training Samples",
    len(X_train)
)

col2.metric(
    "Testing Samples",
    len(X_test)
)

col3.metric(
    "MAE",
    f"${mae:.2f}"
)

col4.metric(
    "RMSE",
    f"${rmse:.2f}"
)


st.write(
    f"""
    The model has a **Mean Absolute Error (MAE) of ${mae:.2f}**.
    This means that, on average, its predicted hourly EC2 price
    differs from the actual price by approximately **${mae:.2f}**.

    The **RMSE is ${rmse:.2f}**. Because RMSE gives more weight
    to large errors, the higher RMSE suggests that some expensive
    EC2 instances have larger prediction errors.
    """
)


# ============================================================
# MODEL COEFFICIENTS
# ============================================================

with st.expander("View Linear Regression Model Details"):

    st.write(
        "Intercept:",
        model.intercept_
    )

    coefficient_df = pd.DataFrame(
        {
            "Feature": [
                "Instance Memory",
                "vCPUs"
            ],
            "Coefficient": model.coef_
        }
    )

    st.dataframe(
        coefficient_df,
        use_container_width=True
    )


# ============================================================
# ACTUAL VS PREDICTED GRAPH
# ============================================================

st.subheader("Actual vs Predicted On-Demand Costs")

fig4, ax4 = plt.subplots(figsize=(8, 6))

ax4.scatter(
    y_test,
    y_pred,
    alpha=0.7
)

ax4.plot(
    [
        min(y_test),
        max(y_test)
    ],
    [
        min(y_test),
        max(y_test)
    ],
    linestyle="--"
)

ax4.set_title(
    "Actual vs Predicted On-Demand Costs"
)

ax4.set_xlabel(
    "Actual On-Demand Cost"
)

ax4.set_ylabel(
    "Predicted On-Demand Cost"
)

plt.tight_layout()

st.pyplot(fig4)

st.caption(
    """
    Points closer to the diagonal line represent more accurate
    predictions. Points farther from the line represent larger
    prediction errors.
    """
)


# ============================================================
# INTERACTIVE EC2 COST PREDICTOR
# ============================================================

# ============================================================
# INTERACTIVE EC2 COST PREDICTOR
# ============================================================

st.header("5. Interactive EC2 Cost Predictor")

st.write(
    """
    Enter the memory and number of vCPUs for a hypothetical
    EC2 instance. The trained Linear Regression model will
    estimate its Linux On-Demand hourly cost.
    """
)

col1, col2 = st.columns(2)

with col1:
    memory_input = st.number_input(
        "Instance Memory (GiB)",
        min_value=0.5,
        value=4.0,
        step=0.5
    )

with col2:
    vcpu_input = st.number_input(
        "Number of vCPUs",
        min_value=1,
        value=2,
        step=1
    )

if st.button(
    "Predict EC2 Cost",
    type="primary",
    key="ec2_predict_button"
):

    new_instance = pd.DataFrame(
        {
            "Instance Memory": [memory_input],
            "vCPUs": [vcpu_input]
        }
    )

    # Make prediction
    raw_prediction = model.predict(new_instance)[0]

    # EC2 prices cannot realistically be negative
    predicted_cost = max(0, raw_prediction)

    st.subheader(
        f"Predicted On-Demand Cost: "
        f"${predicted_cost:.4f} per hour"
    )

    # Calculate approximate monthly cost
    monthly_cost = predicted_cost * 730

    st.write(
        f"Approximate monthly cost "
        f"(730 hours): **${monthly_cost:.2f}**"
    )

    # Show warning if regression produced a negative raw value
    if raw_prediction < 0:
        st.warning(
            """
            The regression model produced a negative raw prediction.
            Since an EC2 price cannot realistically be negative,
            the displayed prediction has been limited to $0.
            """
        )


# ============================================================
# CONCLUSION
# ============================================================

st.header("6. Conclusion")

st.write(
    """
    The analysis shows differences between EC2 pricing models,
    with Reserved and Spot pricing generally offering lower costs
    than On-Demand pricing.

    The Linear Regression model demonstrates how machine learning
    can be used to estimate EC2 On-Demand costs from instance
    characteristics. However, memory and vCPUs alone do not fully
    explain EC2 pricing, especially for high-cost and specialized
    instances.
    """
)