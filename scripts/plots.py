import matplotlib.pyplot as plt

def plot_scatter(df, app_column, total_column='Total Data (Bytes)'):
    plt.figure(figsize=(8, 6))
    plt.scatter(df[app_column], df[total_column], alpha=0.5)
    plt.xlabel(app_column)
    plt.ylabel(total_column)
    plt.title(f'{app_column} vs. {total_column}')
    plt.show()



