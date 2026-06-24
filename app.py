import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


# ============================================================
# Page Configuration
# ============================================================
st.set_page_config(
    page_title="DelayGuard | Delivery Delay Prediction",
    page_icon="📦",
    layout="wide"
)


# ============================================================
# Paths
# ============================================================
BASE_PATH = Path(__file__).resolve().parent

MODELS_PATH = BASE_PATH / "models"
DATA_PATH = BASE_PATH / "Data" / "processed"
RESULTS_PATH = BASE_PATH / "results"

PIPELINE_PATH = MODELS_PATH / "olist_delivery_random_forest_pipeline.joblib"
METADATA_PATH = MODELS_PATH / "app_metadata.joblib"
THRESHOLD_PATH = MODELS_PATH / "classification_threshold.json"

CLEANED_DATA_PATH = DATA_PATH / "olist_delivery_modeling_clean.csv"
FINAL_METRICS_PATH = RESULTS_PATH / "final_test_metrics.csv"
CONFUSION_MATRIX_PATH = RESULTS_PATH / "final_confusion_matrix.csv"


# ============================================================
# Load Artifacts
# ============================================================
@st.cache_resource
def load_model_artifacts():
    pipeline = joblib.load(PIPELINE_PATH)
    metadata = joblib.load(METADATA_PATH)

    with open(THRESHOLD_PATH, "r", encoding="utf-8") as file:
        threshold_data = json.load(file)

    return pipeline, metadata, threshold_data


@st.cache_data
def load_cleaned_data():
    if CLEANED_DATA_PATH.exists():
        return pd.read_csv(
            CLEANED_DATA_PATH,
            parse_dates=[
                "order_purchase_timestamp",
                "order_estimated_delivery_date"
            ]
        )

    return None


@st.cache_data
def load_final_metrics():
    if FINAL_METRICS_PATH.exists():
        return pd.read_csv(FINAL_METRICS_PATH)

    return None


@st.cache_data
def load_confusion_matrix():
    if CONFUSION_MATRIX_PATH.exists():
        return pd.read_csv(
            CONFUSION_MATRIX_PATH,
            index_col=0
        )

    return None


try:
    model_pipeline, app_metadata, threshold_data = load_model_artifacts()

except FileNotFoundError as error:
    st.error(
        "Model files were not found. Please make sure the `models` folder "
        "contains the saved pipeline, metadata, and threshold files."
    )
    st.exception(error)
    st.stop()


modeling_data = load_cleaned_data()
final_metrics = load_final_metrics()
confusion_matrix_df = load_confusion_matrix()

selected_features = app_metadata["selected_features"]
categorical_features = app_metadata["categorical_features"]
numeric_features = app_metadata["numeric_features"]
categorical_options = app_metadata["categorical_options"]
numeric_metadata = app_metadata["numeric_metadata"]
classification_threshold = threshold_data["classification_threshold"]


# ============================================================
# Sidebar Navigation
# ============================================================
st.sidebar.title("📦 DelayGuard")

page = st.sidebar.radio(
    "Navigation",
    [
        "Predict",
        "Model Training Journey",
        "EDA Dashboard"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Model Info")
st.sidebar.write("**Model:** Tuned Random Forest")
st.sidebar.write("**Task:** Binary Classification")
st.sidebar.write("**Target:** Late Delivery")
st.sidebar.write(f"**Threshold:** {classification_threshold:.4f}")


# ============================================================
# Helper Functions
# ============================================================
def number_input_for_feature(feature_name: str):
    metadata = numeric_metadata[feature_name]

    default_value = float(metadata["median"])
    min_value = float(metadata["minimum"])
    max_value = float(metadata["maximum"])

    return st.number_input(
        label=feature_name,
        min_value=0.0 if min_value >= 0 else min_value,
        value=default_value,
        step=1.0,
        help=f"Training range: {min_value:.2f} to {max_value:.2f}"
    )


def add_bar_labels(bars, values, decimals=2):
    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.{decimals}f}",
            ha="center",
            va="bottom",
            fontsize=9
        )


# ============================================================
# Page 1: Prediction
# ============================================================
if page == "Predict":

    st.title("📦 DelayGuard")
    st.subheader("E-Commerce Delivery Delay Prediction System")

    st.write(
        """
        This page predicts whether an e-commerce order is likely to be delivered late
        based on order, payment, product, freight, seller, customer, and time-related features.
        """
    )

    st.info(
        f"Current classification threshold: **{classification_threshold:.4f}**"
    )

    st.markdown("## Enter Order Information")

    with st.form("prediction_form"):

        col1, col2, col3 = st.columns(3)
        user_input = {}

        for index, feature in enumerate(selected_features):
            current_column = [col1, col2, col3][index % 3]

            with current_column:
                if feature in categorical_features:
                    options = categorical_options.get(feature, [])

                    if not options:
                        st.warning(f"No options found for {feature}.")
                        user_input[feature] = ""
                    else:
                        user_input[feature] = st.selectbox(
                            label=feature,
                            options=options
                        )

                elif feature in numeric_features:
                    user_input[feature] = number_input_for_feature(feature)

                else:
                    st.warning(
                        f"Feature `{feature}` was not found in metadata. "
                        "Using 0 as default."
                    )
                    user_input[feature] = 0

        submitted = st.form_submit_button("Predict Delivery Status")

    if submitted:

        input_dataframe = pd.DataFrame(
            [user_input],
            columns=selected_features
        )

        late_probability = model_pipeline.predict_proba(
            input_dataframe
        )[0, 1]

        prediction = int(
            late_probability >= classification_threshold
        )

        st.markdown("## Prediction Result")

        result_col1, result_col2, result_col3 = st.columns(3)

        with result_col1:
            st.metric(
                label="Late Delivery Probability",
                value=f"{late_probability * 100:.2f}%"
            )

        with result_col2:
            st.metric(
                label="Classification Threshold",
                value=f"{classification_threshold:.4f}"
            )

        with result_col3:
            if prediction == 1:
                st.error("Prediction: Likely Late")
            else:
                st.success("Prediction: Likely On Time")

        st.markdown("### Input Used by the Model")
        st.dataframe(input_dataframe, use_container_width=True)

        st.markdown("### Interpretation")

        if prediction == 1:
            st.write(
                """
                The model classified this order as **high risk for late delivery**.
                In a real logistics workflow, this order could be prioritized for monitoring,
                seller follow-up, or proactive customer communication.
                """
            )
        else:
            st.write(
                """
                The model classified this order as **likely to be delivered on time**.
                This does not guarantee on-time delivery, but the order does not appear
                high-risk based on the current input values.
                """
            )


# ============================================================
# Page 2: Model Training Journey
# ============================================================
elif page == "Model Training Journey":

    st.title("🧠 Model Training Journey")

    st.write(
        """
        This page summarizes the full machine-learning workflow followed in the project,
        from raw data preparation to model selection and final evaluation.
        """
    )

    st.markdown("## 1. Problem Definition")

    st.write(
        """
        The project aims to predict whether an e-commerce order will be delivered later
        than its estimated delivery date.

        The target variable is:

        - **0:** Delivered on time
        - **1:** Delivered late
        """
    )

    st.markdown("## 2. Data Cleaning and Preparation")

    st.write(
        """
        The dataset consisted of multiple related tables, including orders, customers,
        payments, products, sellers, and order items.

        Main cleaning steps included:

        - Converting date columns to datetime format.
        - Investigating missing values by order status.
        - Selecting valid delivered orders.
        - Creating the target variable `is_late`.
        - Cleaning product metadata and correcting misspelled column names.
        - Aggregating payments at the order level.
        - Aggregating order items so each row represents one order.
        - Handling invalid product weights and missing payment records.
        """
    )

    st.markdown("## 3. Feature Engineering")

    st.write(
        """
        New features were created to capture operational and logistical complexity:

        - `freight_to_price_ratio`
        - `seller_customer_same_state`
        - `is_multi_item_order`
        - `is_multi_seller_order`

        These features helped represent delivery cost, geography, and order complexity.
        """
    )

    st.markdown("## 4. Feature Selection")

    st.write(
        """
        Mutual Information was used as a filter-based feature-selection technique.
        The top 15 features were selected for modeling.
        """
    )

    st.markdown("### Selected Features")
    st.dataframe(
        pd.DataFrame(
            {
                "selected_feature": selected_features
            }
        ),
        use_container_width=True
    )

    st.markdown("## 5. Model Comparison")

    st.write(
        """
        Three baseline classification models were tested:

        1. Logistic Regression
        2. Decision Tree
        3. Random Forest

        Because the dataset was imbalanced, evaluation focused on precision, recall,
        F1-score, ROC-AUC, and average precision instead of accuracy alone.
        """
    )

    st.markdown("## 6. Hyperparameter and Threshold Tuning")

    st.write(
        """
        Logistic Regression was tuned using GridSearchCV.

        Random Forest was tuned using RandomizedSearchCV and selected as the final model
        because it achieved the best overall validation performance.

        The classification threshold was also tuned to improve the balance between
        precision and recall.
        """
    )

    st.markdown("## 7. Final Evaluation")

    if final_metrics is not None:
        st.dataframe(
            final_metrics,
            use_container_width=True
        )
    else:
        st.warning(
            "Final metrics file was not found. "
            "Make sure `results/final_test_metrics.csv` exists."
        )

    if confusion_matrix_df is not None:
        st.markdown("### Final Confusion Matrix")
        st.dataframe(
            confusion_matrix_df,
            use_container_width=True
        )

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.imshow(confusion_matrix_df.values)

        ax.set_xticks(range(confusion_matrix_df.shape[1]))
        ax.set_yticks(range(confusion_matrix_df.shape[0]))

        ax.set_xticklabels(confusion_matrix_df.columns)
        ax.set_yticklabels(confusion_matrix_df.index)

        for row in range(confusion_matrix_df.shape[0]):
            for column in range(confusion_matrix_df.shape[1]):
                ax.text(
                    column,
                    row,
                    confusion_matrix_df.values[row, column],
                    ha="center",
                    va="center"
                )

        ax.set_title("Final Confusion Matrix")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("Actual Label")
        st.pyplot(fig)

    else:
        st.warning(
            "Confusion matrix file was not found. "
            "Make sure `results/final_confusion_matrix.csv` exists."
        )


# ============================================================
# Page 3: EDA Dashboard
# ============================================================
elif page == "EDA Dashboard":

    st.title("📊 EDA Dashboard")

    if modeling_data is None:
        st.error(
            "Cleaned dataset was not found. Please make sure "
            "`Data/processed/olist_delivery_modeling_clean.csv` exists."
        )
        st.stop()

    st.write(
        """
        This dashboard presents the main exploratory-data-analysis charts used to
        understand delivery delays.
        """
    )

    st.markdown("## Dataset Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Orders", f"{modeling_data.shape[0]:,}")

    with col2:
        st.metric("Features", f"{modeling_data.shape[1]:,}")

    with col3:
        late_rate = modeling_data["is_late"].mean() * 100
        st.metric("Late Orders", f"{late_rate:.2f}%")

    # ------------------------------------------------------------
    # Chart 1: Target Distribution
    # ------------------------------------------------------------
    st.markdown("## 1. Delivery Status Distribution")

    target_summary = (
        modeling_data["is_late"]
        .value_counts()
        .sort_index()
        .rename_axis("is_late")
        .reset_index(name="count")
    )

    target_summary["delivery_status"] = target_summary["is_late"].map(
        {
            0: "On Time",
            1: "Late"
        }
    )

    target_summary["percentage"] = (
        target_summary["count"]
        / target_summary["count"].sum()
        * 100
    ).round(2)

    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(
        target_summary["delivery_status"],
        target_summary["count"]
    )

    ax.set_title("Distribution of Delivery Status")
    ax.set_xlabel("Delivery Status")
    ax.set_ylabel("Number of Orders")

    for bar, percentage in zip(
        bars,
        target_summary["percentage"]
    ):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{percentage:.2f}%",
            ha="center",
            va="bottom"
        )

    st.pyplot(fig)

    st.write(
        """
        The target distribution shows a strong class imbalance, with late orders
        representing a small percentage of total delivered orders.
        """
    )

    # ------------------------------------------------------------
    # Chart 2: Promised Delivery Days Histogram
    # ------------------------------------------------------------
    st.markdown("## 2. Promised Delivery Days Distribution")

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(
        modeling_data["promised_delivery_days"],
        bins=30,
        edgecolor="black"
    )

    mean_value = modeling_data["promised_delivery_days"].mean()
    median_value = modeling_data["promised_delivery_days"].median()

    ax.axvline(
        mean_value,
        linestyle="--",
        linewidth=2,
        label=f"Mean: {mean_value:.2f}"
    )

    ax.axvline(
        median_value,
        linestyle="-",
        linewidth=2,
        label=f"Median: {median_value:.2f}"
    )

    ax.set_title("Distribution of Promised Delivery Days")
    ax.set_xlabel("Promised Delivery Days")
    ax.set_ylabel("Number of Orders")
    ax.legend()

    st.pyplot(fig)

    # ------------------------------------------------------------
    # Chart 3: Monthly Delay Trend
    # ------------------------------------------------------------
    st.markdown("## 3. Monthly Delivery Delay Rate")

    monthly_delay_summary = (
        modeling_data
        .assign(
            purchase_year_month=(
                modeling_data["order_purchase_timestamp"]
                .dt.to_period("M")
                .astype(str)
            )
        )
        .groupby("purchase_year_month")
        .agg(
            total_orders=("order_id", "count"),
            late_orders=("is_late", "sum")
        )
        .reset_index()
    )

    monthly_delay_summary["late_rate_percentage"] = (
        monthly_delay_summary["late_orders"]
        / monthly_delay_summary["total_orders"]
        * 100
    ).round(2)

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(
        monthly_delay_summary["purchase_year_month"],
        monthly_delay_summary["late_rate_percentage"],
        marker="o"
    )

    ax.set_title("Monthly Delivery Delay Rate Over Time")
    ax.set_xlabel("Purchase Year and Month")
    ax.set_ylabel("Late Orders (%)")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.3)

    st.pyplot(fig)

    # ------------------------------------------------------------
    # Chart 4: Price Segment Delay Rate
    # ------------------------------------------------------------
    st.markdown("## 4. Delay Rate by Price Segment")

    price_segment_data = modeling_data.copy()

    price_segment_data["price_segment"] = pd.qcut(
        price_segment_data["total_item_price"],
        q=4,
        labels=[
            "Low",
            "Medium",
            "High",
            "Very High"
        ],
        duplicates="drop"
    )

    price_segment_summary = (
        price_segment_data
        .groupby(
            "price_segment",
            observed=False
        )
        .agg(
            total_orders=("order_id", "count"),
            late_orders=("is_late", "sum"),
            average_order_price=("total_item_price", "mean")
        )
        .reset_index()
    )

    price_segment_summary["late_rate_percentage"] = (
        price_segment_summary["late_orders"]
        / price_segment_summary["total_orders"]
        * 100
    ).round(2)

    fig, ax = plt.subplots(figsize=(9, 5))

    bars = ax.bar(
        price_segment_summary["price_segment"].astype(str),
        price_segment_summary["late_rate_percentage"]
    )

    ax.set_title("Delivery Delay Rate by Order Price Segment")
    ax.set_xlabel("Order Price Segment")
    ax.set_ylabel("Late Orders (%)")

    for bar, rate in zip(
        bars,
        price_segment_summary["late_rate_percentage"]
    ):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{rate:.2f}%",
            ha="center",
            va="bottom"
        )

    st.pyplot(fig)

    # ------------------------------------------------------------
    # Chart 5: Freight vs Price Scatter
    # ------------------------------------------------------------
    st.markdown("## 5. Order Price vs Freight Cost")

    scatter_sample = modeling_data.sample(
        n=min(10000, len(modeling_data)),
        random_state=42
    )

    price_limit = modeling_data["total_item_price"].quantile(0.99)
    freight_limit = modeling_data["total_freight_value"].quantile(0.99)

    filtered_scatter_sample = scatter_sample.loc[
        (scatter_sample["total_item_price"] <= price_limit)
        & (scatter_sample["total_freight_value"] <= freight_limit)
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    for status, label in [
        (0, "On Time"),
        (1, "Late")
    ]:
        status_data = filtered_scatter_sample.loc[
            filtered_scatter_sample["is_late"] == status
        ]

        ax.scatter(
            status_data["total_item_price"],
            status_data["total_freight_value"],
            alpha=0.4,
            label=label
        )

    ax.set_title("Order Price vs Freight Cost by Delivery Status")
    ax.set_xlabel("Total Item Price")
    ax.set_ylabel("Total Freight Cost")
    ax.legend()

    st.pyplot(fig)

    # ------------------------------------------------------------
    # Chart 6: Customer State Delay
    # ------------------------------------------------------------
    st.markdown("## 6. Delivery Delay by Customer State")

    minimum_orders = 500

    customer_state_summary = (
        modeling_data
        .groupby("customer_state")
        .agg(
            total_orders=("order_id", "count"),
            late_orders=("is_late", "sum")
        )
        .reset_index()
    )

    customer_state_summary["late_rate_percentage"] = (
        customer_state_summary["late_orders"]
        / customer_state_summary["total_orders"]
        * 100
    ).round(2)

    reliable_customer_states = (
        customer_state_summary.loc[
            customer_state_summary["total_orders"] >= minimum_orders
        ]
        .sort_values(
            "late_rate_percentage",
            ascending=False
        )
    )

    fig, ax = plt.subplots(figsize=(11, 6))

    bars = ax.bar(
        reliable_customer_states["customer_state"],
        reliable_customer_states["late_rate_percentage"]
    )

    ax.set_title(
        f"Delivery Delay Rate by Customer State "
        f"(At Least {minimum_orders} Orders)"
    )
    ax.set_xlabel("Customer State")
    ax.set_ylabel("Late Orders (%)")
    ax.tick_params(axis="x", rotation=45)

    for bar, rate in zip(
        bars,
        reliable_customer_states["late_rate_percentage"]
    ):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{rate:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8
        )

    st.pyplot(fig)

    # ------------------------------------------------------------
    # Chart 7: Product Category Delay
    # ------------------------------------------------------------
    st.markdown("## 7. Delivery Delay by Product Category")

    minimum_category_orders = 500

    product_category_summary = (
        modeling_data
        .groupby("main_product_category")
        .agg(
            total_orders=("order_id", "count"),
            late_orders=("is_late", "sum"),
            average_weight_g=("total_product_weight_g", "mean"),
            average_freight_value=("total_freight_value", "mean")
        )
        .reset_index()
    )

    product_category_summary["late_rate_percentage"] = (
        product_category_summary["late_orders"]
        / product_category_summary["total_orders"]
        * 100
    ).round(2)

    reliable_categories = (
        product_category_summary.loc[
            product_category_summary["total_orders"]
            >= minimum_category_orders
        ]
        .sort_values(
            "late_rate_percentage",
            ascending=True
        )
    )

    fig, ax = plt.subplots(figsize=(12, 8))

    bars = ax.barh(
        reliable_categories["main_product_category"],
        reliable_categories["late_rate_percentage"]
    )

    ax.set_title(
        "Delivery Delay Rate by Product Category "
        f"(At Least {minimum_category_orders} Orders)"
    )
    ax.set_xlabel("Late Orders (%)")
    ax.set_ylabel("Product Category")

    for bar, rate in zip(
        bars,
        reliable_categories["late_rate_percentage"]
    ):
        ax.text(
            bar.get_width(),
            bar.get_y() + bar.get_height() / 2,
            f" {rate:.1f}%",
            va="center",
            fontsize=8
        )

    st.pyplot(fig)