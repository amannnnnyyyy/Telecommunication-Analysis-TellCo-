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
from calc_stats import *
from space import giveSpace
from data_manip import *

from IPython.display import display
from sklearn.decomposition import PCA

from calc_stats import *
# Load data
data = returnFromDb()

primary_options = {
    'Task 1: User Overview Analysis': ['Sub Tasks', 'EDA'],
    'Task 2: User Engagement Analysis': ['Engagement metrics per user', 'Engagement per App', 'Kmeans'],
    'Task 3: Experience Analytics': ['top, bottom, and most frequent', 'Thoughput and TCP', 'Kmeans clustering'],
    'Task 4: Satisfaction Analysis': ['Engagement score', 'Regression model', 'Model tracking']
}

# Sidebar for primary select box
st.sidebar.title("Teleco")

# Primary select box
selected_category = st.sidebar.selectbox("Select Major Task", options=list(primary_options.keys()))


data = calc_Total(data)
# Secondary select box based on the primary selection
if selected_category:
    secondary_options = primary_options[selected_category]
    selected_option = st.sidebar.selectbox("Select Sub Task", options=secondary_options)

    # Display the selected option
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



        # Aggregating session data per user
        

        user_behavior = data.groupby('MSISDN/Number').agg({
            'Bearer Id': 'count',
            'Dur. (ms)': 'sum', 
            'Social Media Total':'sum',
            'Email Total':'sum',
            'Google Total':'sum',
            'Youtube Total':'sum',
            'Netflix Total':'sum',
            'Gaming Total':'sum',
            'Other Total':'sum',
            'Total DL (Bytes)': 'sum', 
            'Total UL (Bytes)': 'sum'
        }).rename(columns={
            'Bearer Id': 'xDR_sessions', 
            'Dur. (ms)': 'Total_duration', 
            'Total DL (Bytes)': 'Total_DL', 
            'Total UL (Bytes)': 'Total_UL'
        })

        user_behavior['Total_data_volume'] = user_behavior['Total_DL'] + user_behavior['Total_UL']


        user_behavior['Total_data_volume'] = user_behavior['Total_DL'] + user_behavior['Total_UL']
        user_behavior.head()

        # Display the plot using Streamlit's st.pyplot()
        total_download_data = data['Total DL (Bytes)'].sum() / 1024**2
        total_upload_data = data['Total UL (Bytes)'].sum() / 1024**2

        # Display the total download and upload data
        st.write("Total Download Data (MB):", total_download_data)
        st.write("Total Upload Data (MB):", total_upload_data)

        # Create a plot using matplotlib and display it in Streamlit
        labels = ['Total Download', 'Total Upload']
        values = [total_download_data, total_upload_data]

        fig, ax = plt.subplots()
        ax.bar(labels, values, color=['skyblue', 'coral'])
        ax.set_title('Total Download and Upload Data')
        ax.set_ylabel('Data (megabytes)')

        st.pyplot(fig)







        app_totals = {
            'Google': data['Google Total'].sum(),
            'YouTube': data['Youtube Total'].sum(),
            'Netflix': data['Netflix Total'].sum(),
            'Email': data['Email Total'].sum(),
            'Social Media': data['Social Media Total'].sum(),
            'Gaming': data['Gaming Total'].sum(),
            'Other': data['Other Total'].sum()
        }

        app_totals_df = pd.DataFrame(list(app_totals.items()), columns=['Application', 'Total Data (Bytes)'])

        # Plotting using seaborn and matplotlib
        plt.figure(figsize=(12, 6))

        # Plot main applications
        app_totals_df_filtered = app_totals_df[~app_totals_df['Application'].isin(['Gaming', 'Other'])]
        plt.subplot(1, 2, 1)
        sns.barplot(x='Application', y='Total Data (Bytes)', data=app_totals_df_filtered, palette='Blues_d')
        plt.title('Main Applications', fontsize=12)
        plt.ylabel('Total Data (Bytes)')
        plt.xlabel('Application')
        plt.xticks(rotation=45)

        # Plot Gaming and Other separately
        app_totals_df_high_usage = app_totals_df[app_totals_df['Application'].isin(['Gaming', 'Other'])]
        plt.subplot(1, 2, 2)
        sns.barplot(x='Application', y='Total Data (Bytes)', data=app_totals_df_high_usage, palette='Reds')
        plt.title('High Usage Applications (Gaming & Others)', fontsize=12)
        plt.xticks(rotation=45)
        st.write('')
        st.write('')
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.write('Total Data Usage for High Usage Applications:')
        st.pyplot(plt)

    elif selected_option == 'EDA':
        st.write("This is the exploratory data analysis section")
        missing_values_placeholder = st.empty()
        missing_values = data.isnull().sum()
        missing_values_placeholder.table(missing_values)

        # Function to clean missing data
        def clean_data(data):
            # Fill missing categorical columns with mode
            for col in data.select_dtypes(include=['object']).columns:
                data[col].fillna(data[col].mode()[0], inplace=True)

            # Fill missing numerical columns with mean
            for col in data.select_dtypes(include=[int, float]).columns:
                if data[col].notnull().any():
                    data[col].fillna(data[col].mean(), inplace=True)

        # Create a button to clean data and update missing values
        if st.button('Clean Data'):
            clean_data(data)
            st.write("Data cleaned successfully!")
            
            # Display the new missing values table after cleaning
            updated_missing_values = data.isnull().sum()
            missing_values_placeholder.table(updated_missing_values)

        data['Total Data (Bytes)'] = data['Total DL (Bytes)'] + data['Total UL (Bytes)']
        data['Total Duration (s)'] = data['Dur. (ms)'] / 1000

        # Segment the users into deciles based on the total duration
        data['Duration Decile'] = pd.qcut(data['Total Duration (s)'], q=5, labels=False) + 1
        decile_data = data.groupby('Duration Decile')['Total Data (Bytes)'].sum().reset_index()
        decile_data.columns = ['Decile Class', 'Total Data (Bytes)']

        # Create the bar plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(decile_data['Decile Class'], decile_data['Total Data (Bytes)'], color='skyblue')
        ax.set_title('Total Data Usage by Duration Decile Class')
        ax.set_xlabel('Duration Decile Class')
        ax.set_ylabel('Total Data (Bytes)')
        ax.set_xticks(decile_data['Decile Class'])
        ax.grid(True)

        # Display the plot using st.pyplot()
        st.pyplot(fig)

        

        # Select relevant data for PCA
        pca_data = data[['Youtube DL (Bytes)', 'Netflix DL (Bytes)', 'Gaming DL (Bytes)', 'Other DL (Bytes)']]

        # Perform PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(pca_data)

        st.write('')
        st.write('')
        st.write('')
        st.write('Dispersion Parameters: Univariate Non-Graphical')
        dispersion_parameters = compute_dispersion(data)
        st.write(dispersion_parameters)

        
        def plot_dispersion_side_by_side(dispersion_summary, columns_per_row=2):
            num_columns = len(dispersion_summary.columns)
            num_rows = int(np.ceil(num_columns / columns_per_row))

            fig, axes = plt.subplots(num_rows, columns_per_row, figsize=(10 * columns_per_row, 6 * num_rows))
            axes = axes.ravel()

            for i, column in enumerate(dispersion_summary.columns):
                metric_types = dispersion_summary.index

                if i < num_columns:
                    row = i // columns_per_row
                    col = i % columns_per_row

                    for j, metric in enumerate(metric_types):
                        axes[i].plot(dispersion_summary[column])
                        axes[i].set_title(f"{metric} - {column}")
                        axes[i].set_xlabel(column)

                    if row == num_rows - 1:
                        axes[i].set_xlabel(column)

                    axes[i].set_ylabel(metric_types[0])

            fig.suptitle('Dispersion Analysis - Side-by-Side Plots', fontsize=14)
            plt.tight_layout()
            st.pyplot(fig)

        # Usage in Streamlit
        # dispersion_parameters = pd.DataFrame(...)  # Provide your dispersion parameters
        columns_per_row = 3  # Number of columns per row

        plot_dispersion_side_by_side(dispersion_parameters, columns_per_row)


        cols = ['Avg RTT DL (ms)', 'Avg RTT DL (ms)', 'Avg Bearer TP DL (kbps)', 'Avg Bearer TP UL (kbps)']
    # Display dispersion metrics
        dispersion_summary = compute_dispersion(data)
        st.write("Dispersion Summary:")
        st.dataframe(dispersion_summary)

        for column in cols:
            st.write(f"Generating plots for {column}...\n")
            
            plot_histogram(data, column,True)
            plot_boxplot(data, column,True)
            plot_density(data, column,True)
            plot_violin(data, column,True)

        st.write(' ')
        st.write(' ')
        st.write(' ')
        st.write(' ')
        st.title('Bivariate Analysis')
        app_columns = [
            'Social Media DL (Bytes)', 'Social Media UL (Bytes)',
            'Youtube DL (Bytes)', 'Youtube UL (Bytes)',
            'Netflix DL (Bytes)', 'Netflix UL (Bytes)',
            'Google DL (Bytes)', 'Google UL (Bytes)',
            'Email DL (Bytes)', 'Email UL (Bytes)',
            'Gaming DL (Bytes)', 'Gaming UL (Bytes)',
            'Other DL (Bytes)', 'Other UL (Bytes)'
        ]

        # Add the new total column to the list
        app_columns.append('Total Data (Bytes)')

        # Compute correlation matrix
        correlation_matrix = data[app_columns].corr()
        st.markdown(
            """
            <style>
            .streamlit-expanderHeader {
                font-size: 20px;
            }
            .dataframe {
                width: 100% !important;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        # Display the correlation matrix using st.dataframe for a wider view
        st.dataframe(correlation_matrix['Total Data (Bytes)'], use_container_width=True)


        giveSpace()
        st.title('Total Data Usage with Each Application')
        sns.heatmap(correlation_matrix[['Total Data (Bytes)']], annot=True, cmap='viridis')

        plt.title('Correlation of Total Data Usage with Applications')

        st.pyplot(plt)

        giveSpace()
        st.write(data.head(0))
        

        correlation_matrix = data[['Social Media Total',	'Email Total',	'Google Total',	'Youtube Total',	'Netflix Total',	'Gaming Total',	'Other Total']].corr()

        giveSpace()
        st.subheader('Application Correlations with eachother')

        plt.figure(figsize=(10, 8))  # Adjust the figure size as needed
        sns.set(font_scale=1.2)  # Increase the font size
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm',fmt='.4f', linewidths=1, linecolor='yellow')
        plt.title('Correlation Matrix of Application Data Usage')
        st.pyplot(plt)


    agg_data = data.groupby('IMEI').agg({
        'Dur. (ms)': 'sum', # Total session duration
        'Bearer Id': 'count', # Session frequency (count of sessions)
        'Total DL (Bytes)': 'sum', # Total download traffic
        'Total UL (Bytes)': 'sum'  # Total upload traffic
    }).reset_index()

    # Create total traffic column (DL + UL)
    agg_data['Total Traffic (Bytes)'] = agg_data['Total DL (Bytes)'] + agg_data['Total UL (Bytes)']

    if selected_option == 'Engagement metrics per user':
        st.subheader("This is the engagement metrics per user section")

        engagement_metrics = data.groupby('MSISDN/Number').agg({
            'Bearer Id': 'count',    # Number of sessions
            'Dur. (ms)': 'sum',      # Session duration
            'Avg Bearer TP DL (kbps)': 'sum',  # Download traffic
            'Avg Bearer TP UL (kbps)': 'sum',  # Upload traffic
        })

        # Normalize the engagement metrics
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        normalized_metrics = scaler.fit_transform(engagement_metrics)

        # Perform k-means clustering
        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=3, random_state=0)
        engagement_metrics['cluster'] = kmeans.fit_predict(normalized_metrics)

        # Use elbow method to determine optimal k
        wcss = []
        for i in range(1, 11):
            kmeans = KMeans(n_clusters=i, random_state=0)
            kmeans.fit(normalized_metrics)
            wcss.append(kmeans.inertia_)

        giveSpace()
        st.write('Elbow Method K-means Clustering for Engagement Metrics')

        # Plot elbow method
        plt.plot(range(1, 11), wcss)
        plt.title('Elbow Method')
        plt.xlabel('Number of Clusters')
        plt.ylabel('WCSS')
        st.pyplot(plt)


        

        # Sort data by session frequency, duration, and total traffic
        top_customers_by_frequency = agg_data.sort_values(by='Bearer Id', ascending=False).head(10)
        top_customers_by_duration = agg_data.sort_values(by='Dur. (ms)', ascending=False).head(10)
        top_customers_by_traffic = agg_data.sort_values(by='Total Traffic (Bytes)', ascending=False).head(10)

        def highlight_column(s,column):
            return ['background-color: lightgreen' if s.name == column else '' for _ in s]
        

        st.write('Top customers by frequency: \n')
        st.dataframe(top_customers_by_frequency.style.apply(highlight_column,column = 'Bearer Id', axis=0))

        st.write('Top customers by duration: \n')
        st.dataframe(top_customers_by_duration.style.apply(highlight_column,column = 'Dur. (ms)', axis=0))

        st.write('Top customers by traffic: \n')
        st.dataframe(top_customers_by_traffic.style.apply(highlight_column,column = 'Total Traffic (Bytes)', axis=0))
        
    elif selected_option == 'Engagement per App':    
        from sklearn.preprocessing import MinMaxScaler

        # Normalize the engagement metrics
        scaler = MinMaxScaler()
        agg_data[['Bearer Id', 'Dur. (ms)', 'Total Traffic (Bytes)']] = scaler.fit_transform(agg_data[['Bearer Id', 'Dur. (ms)', 'Total Traffic (Bytes)']])

        from sklearn.cluster import KMeans

        # Run K-means clustering with k=3
        kmeans = KMeans(n_clusters=3, random_state=42)
        agg_data['Engagement Cluster'] = kmeans.fit_predict(agg_data[['Bearer Id', 'Dur. (ms)', 'Total Traffic (Bytes)']])

        # Add the cluster labels back to the non-normalized data
        agg_data['Cluster'] = kmeans.labels_

        # Count the number of data points in each cluster
        cluster_counts = agg_data['Engagement Cluster'].value_counts()

        # Create a bar plot
        sns.barplot(x=cluster_counts.index, y=cluster_counts)

        # Set labels and title
        plt.xlabel('Cluster')
        plt.ylabel('Count')
        plt.title('Cluster Data Count')

        st.pyplot(plt)


        agg_data[['Bearer Id', 'Dur. (ms)', 'Total Traffic (Bytes)']] = scaler.inverse_transform(
            agg_data[['Bearer Id', 'Dur. (ms)', 'Total Traffic (Bytes)']])

        cluster_summary = agg_data.groupby('Cluster').agg({
            'Bearer Id': ['min', 'max', 'mean', 'sum'],
            'Dur. (ms)': ['min', 'max', 'mean', 'sum'],
            'Total Traffic (Bytes)': ['min', 'max', 'mean', 'sum']
        })

        st.write(cluster_summary)
        giveSpace()
        # Get the top 10 most engaged users per application
        top_social_media_users = data[['IMEI', 'Social Media Total']].nlargest(10, 'Social Media Total')
        top_youtube_users = data[['IMEI', 'Youtube Total']].nlargest(10, 'Youtube Total')
        top_netflix_users = data[['IMEI', 'Netflix Total']].nlargest(10, 'Netflix Total')

        # Display the top users per application
        st.write("Top 10 Most Engaged Users for Social Media:")
        st.write(top_social_media_users)
        st.write("\nTop 10 Most Engaged Users for YouTube:")
        st.write(top_youtube_users)
        st.write("\nTop 10 Most Engaged Users for Netflix:")
        st.write(top_netflix_users)
        giveSpace()
        # Get the top 10 most engaged users per application
        top_users_per_app = data[['IMEI', 'Social Media Total', 'Youtube Total', 'Netflix Total','Gaming Total', 'Other Total']].nlargest(10, 'Social Media Total')
        st.write(top_users_per_app)

        giveSpace()
        app_totals = pd.DataFrame({
            'Application': ['Social Media', 'YouTube', 'Netflix', 'Google', 'Email', 'Gaming', 'Other'],
            'Total Usage (Bytes)': [
                data['Social Media Total'].sum(),
                data['Youtube Total'].sum(),
                data['Netflix Total'].sum(),
                data['Google Total'].sum(),
                data['Email Total'].sum(),
                data['Gaming Total'].sum(),
                data['Other Total'].sum()
            ]
        })

        # Step 3: Find the top 3 most used applications
        top_3_apps = app_totals.nlargest(3, 'Total Usage (Bytes)')
        st.write("Top 3 Most Used Applications:")
        st.write(top_3_apps)

        # Plotting the top 3 most used applications
        apps = ['Gaming Total', 'Youtube Total', 'Other Total']
        data[apps].sum().plot(kind='bar', title='Top 3 Most Used Applications')
        plt.ylabel('Total Data Usage (Bytes)')
        st.pyplot(plt)

        giveSpace()

    elif selected_option == 'Kmeans':
        from sklearn.metrics import silhouette_score

        inertia = []
        K = range(1, 10)
        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(agg_data[['Bearer Id', 'Dur. (ms)', 'Total Traffic (Bytes)']])
            inertia.append(kmeans.inertia_)

        # Plot the elbow curve
        plt.plot(K, inertia, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Inertia')
        plt.title('Elbow Method to find optimal k')
        st.pyplot(plt)

        giveSpace()

    data['TCP DL Retrans. Vol (Bytes)'].fillna(data['TCP DL Retrans. Vol (Bytes)'].mean(), inplace=True)
    data['Avg RTT DL (ms)'].fillna(data['Avg RTT DL (ms)'].mean(), inplace=True)
    data['Avg Bearer TP DL (kbps)'].fillna(data['Avg Bearer TP DL (kbps)'].mean(), inplace=True)
    data['Handset Type'].fillna(data['Handset Type'].mode()[0], inplace=True)

    customer_data = data.groupby('MSISDN/Number').agg({
        'TCP DL Retrans. Vol (Bytes)': 'mean',
        'Avg RTT DL (ms)': 'mean',
        'Avg Bearer TP DL (kbps)': 'mean',
        'Handset Type': lambda x: x.mode()[0]  # Mode to capture most frequent handset type
    }).reset_index()

    customer_data.columns = ['MSISDN', 'Avg TCP Retrans', 'Avg RTT', 'Avg Throughput', 'Handset Type']

        # Top 10 TCP retransmission
    top_10_tcp = customer_data.nlargest(10, 'Avg TCP Retrans')

    # Bottom 10 TCP retransmission
    bottom_10_tcp = customer_data.nsmallest(10, 'Avg TCP Retrans')

    # Most frequent TCP values
    most_frequent_tcp = customer_data['Avg TCP Retrans'].mode()

    top_10_rtt = customer_data.nlargest(10, 'Avg RTT')

    # Bottom 10 RTT
    bottom_10_rtt = customer_data.nsmallest(10, 'Avg RTT')

    # Most frequent RTT values
    most_frequent_rtt = customer_data['Avg RTT'].mode()


    # Top 10 Throughput
    top_10_throughput = customer_data.nlargest(10, 'Avg Throughput')

    # Bottom 10 Throughput
    bottom_10_throughput = customer_data.nsmallest(10, 'Avg Throughput')

    # Most frequent Throughput values
    most_frequent_throughput = customer_data['Avg Throughput'].mode()


    # Step 1: Prepare the data for Top, Bottom, and Most Frequent Throughput
    top_throughput = customer_data.groupby('Handset Type')['Avg Throughput'].mean().reset_index()
    top_throughput = top_throughput.sort_values(by='Avg Throughput', ascending=False).head(10)
    top_throughput['Category'] = 'Top 10'

    bottom_throughput = customer_data.groupby('Handset Type')['Avg Throughput'].mean().reset_index()
    bottom_throughput = bottom_throughput.sort_values(by='Avg Throughput', ascending=True).head(10)
    bottom_throughput['Category'] = 'Bottom 10'

    most_frequent = customer_data['Handset Type'].value_counts().reset_index().head(10)
    most_frequent.columns = ['Handset Type', 'Count']
    most_frequent = pd.merge(most_frequent, customer_data.groupby('Handset Type')['Avg Throughput'].mean().reset_index(), on='Handset Type')
    most_frequent['Category'] = 'Most Frequent'

    # Combine data into a single DataFrame
    combined_throughput = pd.concat([top_throughput, bottom_throughput, most_frequent[['Handset Type', 'Avg Throughput', 'Category']]])

    # Step 1: Extract top, bottom, and most frequent data for TCP Retransmission, RTT, and Throughput
    top_tcp = customer_data.groupby('Handset Type')['Avg TCP Retrans'].mean().reset_index().sort_values(by='Avg TCP Retrans', ascending=False).head(10)
    bottom_tcp = customer_data.groupby('Handset Type')['Avg TCP Retrans'].mean().reset_index().sort_values(by='Avg TCP Retrans', ascending=True).head(10)
    most_frequent_tcp = customer_data['Handset Type'].value_counts().reset_index().head(10)
    most_frequent_tcp.columns = ['Handset Type', 'Count']
    most_frequent_tcp = pd.merge(most_frequent_tcp, customer_data.groupby('Handset Type')['Avg TCP Retrans'].mean().reset_index(), on='Handset Type')

    top_rtt = customer_data.groupby('Handset Type')['Avg RTT'].mean().reset_index().sort_values(by='Avg RTT', ascending=False).head(10)
    bottom_rtt = customer_data.groupby('Handset Type')['Avg RTT'].mean().reset_index().sort_values(by='Avg RTT', ascending=True).head(10)
    most_frequent_rtt = pd.merge(most_frequent_tcp[['Handset Type', 'Count']], customer_data.groupby('Handset Type')['Avg RTT'].mean().reset_index(), on='Handset Type')

    top_throughput = customer_data.groupby('Handset Type')['Avg Throughput'].mean().reset_index().sort_values(by='Avg Throughput', ascending=False).head(10)
    bottom_throughput = customer_data.groupby('Handset Type')['Avg Throughput'].mean().reset_index().sort_values(by='Avg Throughput', ascending=True).head(10)
    most_frequent_throughput = pd.merge(most_frequent_tcp[['Handset Type', 'Count']], customer_data.groupby('Handset Type')['Avg Throughput'].mean().reset_index(), on='Handset Type')


    combined_tcp = pd.concat([top_tcp, bottom_tcp, most_frequent_tcp[['Handset Type', 'Avg TCP Retrans']]])
    combined_tcp['Category'] = ['Top 10']*10 + ['Bottom 10']*10 + ['Most Frequent']*10

    combined_rtt = pd.concat([top_rtt, bottom_rtt, most_frequent_rtt[['Handset Type', 'Avg RTT']]])
    combined_rtt['Category'] = ['Top 10']*10 + ['Bottom 10']*10 + ['Most Frequent']*10

    combined_throughput = pd.concat([top_throughput, bottom_throughput, most_frequent_throughput[['Handset Type', 'Avg Throughput']]])
    combined_throughput['Category'] = ['Top 10']*10 + ['Bottom 10']*10 + ['Most Frequent']*10

    if selected_option == 'top, bottom, and most frequent':
        st.title('Top Bottom and Most Frequent')
        # Step 3: Create subplots for TCP, RTT, and Throughput
        fig, axes = plt.subplots(3, 1, figsize=(12, 18))

        # Plot TCP Retransmission
        sns.barplot(x='Avg TCP Retrans', y='Handset Type', hue='Category', data=combined_tcp, ax=axes[0], palette='muted')
        axes[0].set_title('Top, Bottom, and Most Frequent Handset Types by TCP Retransmission')
        axes[0].set_xscale('log')  # Apply log scale for visibility
        axes[0].set_xlabel('Average TCP Retransmission (log scale)')

        # Plot RTT
        sns.barplot(x='Avg RTT', y='Handset Type', hue='Category', data=combined_rtt, ax=axes[1], palette='muted')
        axes[1].set_title('Top, Bottom, and Most Frequent Handset Types by RTT')
        axes[1].set_xscale('log')  # Apply log scale for visibility
        axes[1].set_xlabel('Average RTT (log scale)')

        # Plot Throughput
        sns.barplot(x='Avg Throughput', y='Handset Type', hue='Category', data=combined_throughput, ax=axes[2], palette='muted')
        axes[2].set_title('Top, Bottom, and Most Frequent Handset Types by Throughput')
        axes[2].set_xscale('log')  # Apply log scale for visibility
        axes[2].set_xlabel('Average Throughput (log scale)')

        # Adjust layout and show plot
        plt.tight_layout()
        st.pyplot(plt)

    if selected_option == 'Thoughput and TCP':
        # Sort by average throughput and select top 15 handset types for better visibility
        throughput_dist = customer_data.groupby('Handset Type')['Avg Throughput'].mean().reset_index()
        throughput_dist = throughput_dist.sort_values(by='Avg Throughput', ascending=False).head(15)
    
        # Larger figure, distinct color palette, and sorted bar plot
        plt.figure(figsize=(14, 8))
        sns.barplot(x='Avg Throughput', y='Handset Type', data=throughput_dist, palette="coolwarm")

        plt.title('Top 15 Handset Types by Average Throughput', fontsize=16)
        plt.xlabel('Average Throughput (kbps)', fontsize=12)
        plt.ylabel('Handset Type', fontsize=12)
        plt.tight_layout()
        st.pyplot(plt)


        # Sort by average TCP retransmission and select top 15 handset types
        tcp_dist = customer_data.groupby('Handset Type')['Avg TCP Retrans'].mean().reset_index()
        tcp_dist = tcp_dist.sort_values(by='Avg TCP Retrans', ascending=False).head(15)

        # Larger figure, distinct color palette, and sorted bar plot
        plt.figure(figsize=(14, 8))
        sns.barplot(x='Avg TCP Retrans', y='Handset Type', data=tcp_dist, palette="coolwarm")

        plt.title('Top 15 Handset Types by Average TCP Retransmission', fontsize=16)
        plt.xlabel('Average TCP Retransmission (Bytes)', fontsize=12)
        plt.ylabel('Handset Type', fontsize=12)
        plt.tight_layout()
        st.pyplot(plt)

    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans

    agg_data = data.groupby('MSISDN/Number').agg({
        'TCP DL Retrans. Vol (Bytes)': 'mean',
        'Avg RTT DL (ms)': 'mean',
        'Avg Bearer TP DL (kbps)': 'mean'
    }).reset_index()

    # Rename columns for convenience
    agg_data.rename(columns={
        'TCP DL Retrans. Vol (Bytes)': 'Avg_TCP_Retrans',
        'Avg RTT DL (ms)': 'Avg_RTT',
        'Avg Bearer TP DL (kbps)': 'Avg_Throughput'
    }, inplace=True)

    # Step 2: Normalize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(agg_data[['Avg_TCP_Retrans', 'Avg_RTT', 'Avg_Throughput']])

    if selected_option == 'Kmeans clustering':
        kmeans = KMeans(n_clusters=3, random_state=0)
        agg_data['Cluster'] = kmeans.fit_predict(scaled_data)
        cluster_summary = agg_data.groupby('Cluster').agg({
            'Avg_TCP_Retrans': ['mean', 'min', 'max'],
            'Avg_RTT': ['mean', 'min', 'max'],
            'Avg_Throughput': ['mean', 'min', 'max']
        })

        st.write(cluster_summary)

        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Scatter plot
        scatter = ax.scatter(
            agg_data['Avg_TCP_Retrans'],
            agg_data['Avg_RTT'],
            agg_data['Avg_Throughput'],
            c=agg_data['Cluster'],  # Color by cluster
            cmap='viridis',  # Color map
            s=50,  # Marker size
            alpha=0.6
        )

        # Labels
        ax.set_xlabel('Average TCP Retransmission (Bytes)')
        ax.set_ylabel('Average RTT (ms)')
        ax.set_zlabel('Average Throughput (kbps)')
        plt.title('3D Scatter Plot of Clusters', fontsize=16)
        plt.colorbar(scatter, label='Cluster')
        st.pyplot(plt)


        # Plot pairwise relationships
        sns.pairplot(agg_data, hue='Cluster', palette='viridis', vars=['Avg_TCP_Retrans', 'Avg_RTT', 'Avg_Throughput'])
        plt.suptitle('Pairwise Plots of Clusters', y=1.02, fontsize=16)
        st.pyplot(plt)

        # Function to plot radar chart
        def plot_radar(df, title, labels, colors):
            num_vars = len(labels)
            angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
            angles += angles[:1]

            fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
            for i, color in enumerate(colors):
                values = df.loc[i].values.flatten().tolist()
                values += values[:1]
                ax.plot(angles, values, color=color, linewidth=2, linestyle='solid', label=f'Cluster {i}')
                ax.fill(angles, values, color=color, alpha=0.25)

            ax.set_yticklabels([])
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(labels)
            plt.title(title, fontsize=16)
            plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
            st.pyplot(plt)

        # Prepare data for radar chart
        cluster_means = agg_data.groupby('Cluster').mean().reset_index()
        labels = ['Avg_TCP_Retrans', 'Avg_RTT', 'Avg_Throughput']
        colors = ['blue', 'green', 'red']
        plot_radar(cluster_means[labels], 'Radar Chart of Clusters', labels, colors)

    features = ['Avg_TCP_Retrans', 'Avg_RTT', 'Avg_Throughput']
    X = agg_data[features]

    
    

    # Perform K-Means clustering for engagement
    kmeans_engagement = KMeans(n_clusters=3, random_state=42)
    agg_data['Engagement_Cluster'] = kmeans_engagement.fit_predict(X)
    engagement_centroids = kmeans_engagement.cluster_centers_

    # Perform K-Means clustering for experience
    kmeans_experience = KMeans(n_clusters=3, random_state=42)
    agg_data['Experience_Cluster'] = kmeans_experience.fit_predict(X)
    experience_centroids = kmeans_experience.cluster_centers_

    # Compute engagement and experience scores
    from model_related_calc import *
    least_engaged_centroid = engagement_centroids[agg_data['Engagement_Cluster'].mode().iloc[0]]
    worst_experience_centroid = experience_centroids[agg_data['Experience_Cluster'].mode().iloc[0]]

    agg_data['Engagement Score'] = compute_distance(agg_data, least_engaged_centroid,features)
    agg_data['Experience Score'] = compute_distance(agg_data, worst_experience_centroid,features)

    agg_data.to_csv('engagement_experience_scores.csv', index=False)

    scores_data = pd.read_csv('engagement_experience_scores.csv')

    # Calculate satisfaction score
    scores_data['Satisfaction Score'] = (scores_data['Engagement Score'] + scores_data['Experience Score']) / 2

    # Get top 10 satisfied customers
    top_10_satisfied = scores_data.nlargest(10, 'Satisfaction Score')

    if selected_option == 'Engagement score':
        st.title('Engagement Score')
        # Display top 10 satisfied customers
        st.write(top_10_satisfied[['MSISDN/Number', 'Satisfaction Score']])


    if selected_option == 'Regression model':
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error

        # Prepare data for regression model
        features = ['Engagement Score', 'Experience Score']
        X = scores_data[features]
        y = scores_data['Satisfaction Score']

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Train regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predict and evaluate
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)

        # Display results
        st.write(f"Mean Squared Error of the regression model: {mse}")


        # Prepare data for K-Means clustering
        X_scores = scores_data[['Engagement Score', 'Experience Score']]

        # Perform K-Means clustering with k=2
        kmeans_scores = KMeans(n_clusters=2, random_state=42)
        scores_data['Cluster'] = kmeans_scores.fit_predict(X_scores)

        # Display clustering results
        st.write(scores_data[['MSISDN/Number', 'Cluster']])

        # Aggregate average satisfaction and experience scores per cluster
        cluster_aggregates = scores_data.groupby('Cluster').agg({
            'Satisfaction Score': 'mean',
            'Experience Score': 'mean'
        }).reset_index()

        # Display aggregated scores
        st.write(cluster_aggregates)

        
