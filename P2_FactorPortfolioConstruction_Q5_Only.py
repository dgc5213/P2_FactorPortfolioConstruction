import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
def readCSV(path_csv):
    df = pd.read_csv(path_csv)
    return df

def prepareData(df):
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    print ("prepareData----end")
    return df


if __name__ == '__main__':
    print ("---------------------start ---------------\n")


###----------!!!!----Please edit the raw data path and choose correct file format----------------------
    raw_csv = "1_Rawdata/p2_class_project_data.csv"
    df_raw = readCSV(raw_csv)
    # print(df_raw)
    df=df_raw
    df=prepareData(df)



    ##### N5: Given this limitation, show how you might do a sanity check that the P/B value is indeed the ratio between market cap and book value?
    # Step 1: Calculate the self-calculated P/B value
    df['self_calculated_PB'] = df['Mkt_Cap_3M_Usd'] / df['Bv']

    # Step 2: Calculate the correlation between self-calculated P/B and P/B
    correlation = df['self_calculated_PB'].corr(df['Pb'])

    # Step 3: Visualize the scatter plot
    plt.scatter(df['Pb'], df['self_calculated_PB'])
    plt.xlabel('Price-to-Book Ratio (P/B)')
    plt.ylabel('Self-Calculated P/B')
    plt.title(f'Correlation: {correlation:.2f}')
    plt.savefig('2_IMG/Correlation.png')
    # plt.show()
    plt.pause(1)
    plt.close()


    ## identify outliers by specifying a threshold for correlation
    absolute_difference = np.abs(df['self_calculated_PB'] - df['Pb'])
    threshold = 5 # may adjust this value
    outliers = df[absolute_difference > threshold]
    print('Outliers:')
    print(outliers)

    ##Visualize the scatter plot and outliers
    plt.scatter(df['Pb'], df['self_calculated_PB'])
    plt.xlabel('Price-to-Book Ratio (P/B)')
    plt.ylabel('Self-Calculated P/B')
    plt.title(f'Scatter Plot with Outliers Highlighted,Threshold: {threshold:.2f}')
    ## Plot outliers in red
    plt.scatter(outliers['Pb'], outliers['self_calculated_PB'], c='red', label='Outliers')
    plt.legend()
    plt.savefig('2_IMG/Correlation_outliers.png')
    # plt.show()
    plt.pause(1)
    plt.close()