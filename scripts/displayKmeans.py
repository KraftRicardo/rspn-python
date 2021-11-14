import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def display(filepath: str, graphTitle: str, numColors: int):
    plt.figure()
    df = pd.read_csv(filepath)
    sns.scatterplot(x=df.x, y=df.y, hue=df.c, palette=sns.color_palette("hls", n_colors=numColors))
    plt.xlabel("Annual income (k$)")
    plt.ylabel("Spending Score (1-100)")
    plt.title("Clustered: spending (y) vs income (x) " + graphTitle)
    plt.show()

if __name__ == '__main__':
    print("Starting Display ...")

    display("res/Mall_customer_kmeans.csv", "5 iterations", 3);
    display("res/Mall_customer_kmeans_10.csv", "10 iterations", 3);
    display("res/Mall_customer_kmeans_res1.csv", "res1", 3);
    display("res/Mall_customer_kmeans_5i_6c.csv", "5 iterations", 6);
    display("res/Mall_customer_kmeans_10i_6c.csv", "10 iterations", 6);
