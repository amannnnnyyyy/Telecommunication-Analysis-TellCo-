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
        data['Social Media Total'] = data['Social Media DL (Bytes)'] + data['Social Media UL (Bytes)']
        data['Email Total'] = data['Email UL (Bytes)']+data['Email DL (Bytes)']
        data['Google Total'] = data['Google DL (Bytes)'] + data['Google UL (Bytes)']
        data['Youtube Total'] = data['Youtube DL (Bytes)'] + data['Youtube UL (Bytes)']
        data['Netflix Total'] = data['Netflix DL (Bytes)'] + data['Netflix UL (Bytes)']
        data['Gaming Total'] = data['Gaming DL (Bytes)']+data['Gaming UL (Bytes)']
        data['Other Total'] = data['Other DL (Bytes)'] + data['Other UL (Bytes)']

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
        data['Social Media Total'] = data['Social Media DL (Bytes)'] + data['Social Media UL (Bytes)']
        data['Email Total'] = data['Email UL (Bytes)']+data['Email DL (Bytes)']
        data['Google Total'] = data['Google DL (Bytes)'] + data['Google UL (Bytes)']
        data['Youtube Total'] = data['Youtube DL (Bytes)'] + data['Youtube UL (Bytes)']
        data['Netflix Total'] = data['Netflix DL (Bytes)'] + data['Netflix UL (Bytes)']
        data['Gaming Total'] = data['Gaming DL (Bytes)']+data['Gaming UL (Bytes)']
        data['Other Total'] = data['Other DL (Bytes)'] + data['Other UL (Bytes)']

        correlation_matrix = data[['Social Media Total',	'Email Total',	'Google Total',	'Youtube Total',	'Netflix Total',	'Gaming Total',	'Other Total']].corr()

        giveSpace()
        st.subheader('Application Correlations with eachother')

        plt.figure(figsize=(10, 8))  # Adjust the figure size as needed
        sns.set(font_scale=1.2)  # Increase the font size
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm',fmt='.4f', linewidths=1, linecolor='yellow')
        plt.title('Correlation Matrix of Application Data Usage')
        st.pyplot(plt)

        