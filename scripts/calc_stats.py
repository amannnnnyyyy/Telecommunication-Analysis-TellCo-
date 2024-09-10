import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

def compute_dispersion(df):
    dispersion_summary = pd.DataFrame()

    for column in df.select_dtypes(include=['float64', 'int64']).columns:
        data = df[column]
        
        range_value = data.max() - data.min()
        variance = data.var()
        std_dev = data.std()
        iqr = data.quantile(0.75) - data.quantile(0.25)
        
        dispersion_summary[column] = [range_value, variance, std_dev, iqr]

    dispersion_summary.index = ['Range', 'Variance', 'Standard Deviation', 'Interquartile Range']
    
    return dispersion_summary



def plot_histogram(df, column, streamlit = False):
    plt.figure(figsize=(10, 6))
    df[column].hist(bins=30, edgecolor='black')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.title(f'Histogram of {column}')
    if(streamlit):
        st.pyplot(plt)
    else:
        plt.show()

def plot_boxplot(df, column, streamlit=False):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df[column])
    plt.xlabel(column)
    plt.title(f'Box Plot of {column}')
    if(streamlit):
        st.pyplot(plt)
    else:
        plt.show()

def plot_density(df, column, streamlit=False):
    plt.figure(figsize=(10, 6))
    sns.kdeplot(df[column], shade=True)
    plt.xlabel(column)
    plt.title(f'Density Plot of {column}')
    if(streamlit):
        st.pyplot(plt)
    else:
        plt.show()
def plot_violin(df, column, streamlit = False):
    plt.figure(figsize=(10, 6))
    sns.violinplot(x=df[column])
    plt.xlabel(column)
    plt.title(f'Violin Plot of {column}')
    if(streamlit):
        st.pyplot(plt)
    else:
        plt.show()
