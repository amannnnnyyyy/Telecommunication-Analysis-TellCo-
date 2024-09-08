import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import os
import sys
current_directory = os.path.dirname(os.path.abspath(__file__))
scripts_directory = os.path.join(current_directory, '..', 'scripts')
sys.path.append(scripts_directory)
from db import returnFromDb
from IPython.display import display
# Load data
data = returnFromDb()

primary_options = {
    'Task 1: User Overview Analysis': ['Sub Tasks', "User's behaviour", 'EDA'],
    'Task 2: User Engagement Analysis': ['Engagement metrics per user', 'Engagement per App', 'Kmeans'],
    'Task 3: Experience Analytics': ['top, bottom, and most frequent', 'Thoughput and TCP', 'Kmeans clustering'],
    'Task 4: Satisfaction Analysis': ['Engagement score', 'Regression model', 'Model tracking']
}

# Sidebar for primary select box
st.sidebar.title("Dynamic Select Boxes")

# Primary select box
selected_category = st.sidebar.selectbox("Select Category", options=list(primary_options.keys()))

# Secondary select box based on the primary selection
if selected_category:
    secondary_options = primary_options[selected_category]
    selected_option = st.sidebar.selectbox("Select Option", options=secondary_options)

    # Display the selected option
    st.write(f"You selected: {selected_option}")
    if selected_option == 'Sub Tasks':
        st.write("head: ",data.head())
        st.write("info: This is the raw data imported from database")
        st.write("description stat: \n",data.describe())

        top_10_handsets = data['Handset Type'].value_counts().head(10)
        st.write("Top 10 Handsets:")
        st.write(top_10_handsets)
        fig, ax = plt.subplots()
        top_10_handsets.plot(kind='barh', color='skyblue', ax=ax)
        plt.title('Top 10 Handsets Used by Customers Visually')
        plt.xlabel('Count')
        plt.ylabel('Handset')

# Display the plot using Streamlit's st.pyplot()
        st.pyplot(fig)

        theUndefined = data[data['Handset Type'] == 'undefined']
        st.write("Rows where 'Handset Type' is 'undefined':")
        st.write(theUndefined.head())
        top_3_manufacturers = data['Handset Manufacturer'].value_counts().head(3)

        fig, ax = plt.subplots()
        top_3_manufacturers.plot(kind='bar', color='lightgreen', ax=ax)
        plt.title('Top 3 Handset Manufacturers')
        plt.xlabel('Manufacturers')
        plt.ylabel('Count')
        st.pyplot(fig)


        for manufacturer in top_3_manufacturers.index:
            st.write(f"Top 5 handsets for {manufacturer}:")
            top_5_handsets = data[data['Handset Manufacturer'] == manufacturer]['Handset Type'].value_counts().head(5)
            st.write(top_5_handsets)  # Display the top 5 handsets for the current manufacturer

            # Plotting the top 5 handsets for the current manufacturer as a bar plot
            fig, ax = plt.subplots()
            top_5_handsets.plot(kind='bar', color=['darkgreen', 'blue', 'lightgreen', 'maroon', 'yellow'], ax=ax)
            plt.title(f'Top 5 Handsets for {manufacturer}')
            plt.xlabel('Handsets')
            plt.ylabel('Count')

            # Display the plot for the current manufacturer using Streamlit's st.pyplot()
            st.pyplot(fig)

        # Display the plot using Streamlit's st.pyplot()

# # Sidebar for user input
# st.sidebar.title("Dashboard Navigation")
# selected_task = st.sidebar.selectbox("Choose Task", ["Task 3 - Experience Analytics", "Task 4 - Satisfaction Analysis"])

# if selected_task == "Task 3 - Experience Analytics":
#     st.title("Experience Analytics Dashboard")

#     # Task 3.1: Aggregate data
#     st.header("Task 3.1 - Aggregated Data")
#     st.write(agg_data[['MSISDN/Number', 'Avg_TCP_Retrans', 'Avg_RTT', 'Avg_Throughput', 'Cluster']])

#     # Task 3.2: Top, bottom, and most frequent values
#     st.header("Task 3.2 - Top, Bottom, and Most Frequent Values")
    
#     def display_top_bottom_most_frequent(df, col):
#         top = df[col].nlargest(10)
#         bottom = df[col].nsmallest(10)
#         most_frequent = df[col].mode().head(10)  # Most frequent values
        
#         st.write(f"Top 10 {col}:")
#         st.write(top)
#         st.write(f"Bottom 10 {col}:")
#         st.write(bottom)
#         st.write(f"Most Frequent {col}:")
#         st.write(most_frequent)

#     for col in ['Avg_TCP_Retrans', 'Avg_RTT', 'Avg_Throughput']:
#         display_top_bottom_most_frequent(agg_data, col)

#     # Task 3.3: Distribution of throughput and TCP retransmission by handset type
#     st.header("Task 3.3 - Distribution Analysis")
    
#     for col in ['Avg_Throughput', 'Avg_TCP_Retrans']:
#         st.subheader(f"Distribution of {col} by Handset Type")
#         plt.figure(figsize=(12, 6))
#         sns.boxplot(x='Handset Type', y=col, data=agg_data)
#         plt.xticks(rotation=45)
#         plt.title(f"Distribution of {col} by Handset Type")
#         st.pyplot(plt)

# elif selected_task == "Task 4 - Satisfaction Analysis":
#     st.title("Satisfaction Analysis Dashboard")

#     # Load additional data if needed
#     satisfaction_data = pd.read_csv('satisfaction_data.csv')

#     # Task 4.1: Compute engagement and experience scores
#     st.header("Task 4.1 - Engagement and Experience Scores")
#     st.write(satisfaction_data[['MSISDN/Number', 'Engagement Score', 'Experience Score']])

#     # Task 4.2: Top 10 satisfied customers
#     st.header("Task 4.2 - Top 10 Satisfied Customers")
#     top_10_satisfied = satisfaction_data.nlargest(10, 'Satisfaction Score')
#     st.write(top_10_satisfied[['MSISDN/Number', 'Satisfaction Score']])

#     # Task 4.3: Build and display regression model
#     st.header("Task 4.3 - Regression Model")
#     from sklearn.linear_model import LinearRegression
#     from sklearn.model_selection import train_test_split
#     from sklearn.metrics import mean_squared_error
    
#     features = ['Engagement Score', 'Experience Score']
#     X = satisfaction_data[features]
#     y = satisfaction_data['Satisfaction Score']
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#     model = LinearRegression()
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     mse = mean_squared_error(y_test, y_pred)
#     st.write(f"Mean Squared Error of the regression model: {mse}")

#     # Task 4.4: K-Means clustering on scores
#     st.header("Task 4.4 - K-Means Clustering")
#     kmeans = KMeans(n_clusters=2, random_state=42)
#     satisfaction_data['Cluster'] = kmeans.fit_predict(satisfaction_data[['Engagement Score', 'Experience Score']])
#     st.write(satisfaction_data[['MSISDN/Number', 'Cluster']])

#     # Task 4.5: Aggregate satisfaction and experience scores per cluster
#     st.header("Task 4.5 - Aggregate Scores per Cluster")
#     cluster_aggregates = satisfaction_data.groupby('Cluster').agg({
#         'Satisfaction Score': 'mean',
#         'Experience Score': 'mean'
#     }).reset_index()
#     st.write(cluster_aggregates)

#     # Task 4.6: Export table to MySQL (demonstration only, use appropriate database connection)
#     st.header("Task 4.6 - Export to MySQL")
#     st.write("Ensure your MySQL database is properly set up and connect using SQLAlchemy or similar library.")
    
#     # Task 4.7: Model deployment tracking (not applicable in Streamlit directly)
#     st.header("Task 4.7 - Model Deployment Tracking")
#     st.write("Track model deployment using Docker or MLOps tools.")
