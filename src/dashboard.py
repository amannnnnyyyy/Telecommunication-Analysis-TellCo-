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

