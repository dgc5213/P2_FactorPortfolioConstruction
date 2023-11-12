import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
### Read raw data, and store as data frame
def readCSV(path_csv):
   # df = pd.read_csv(path_csv,nrows=9999) ## to testing to use, sample data
    df = pd.read_csv(path_csv)
    return df


def readExcel(path_xlsx):
    df = pd.read_excel(path_xlsx)
    return df



def prepareData(df):
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    print ("prepareData----end")
    return df

def getPlot_MonthlyStocks(df):


    # Count unique stock_ids
    monthly_stock_counts = df.groupby(['year', 'month'])['stock_id'].nunique()

    # Print monthly stock counts per year-month
    for year_month, count in monthly_stock_counts.items():
        print(f"Year-Month: {year_month}, Stock Count: {count}")

    ### Plot the number of stocks included each month using a bar chart
    monthly_stock_counts = monthly_stock_counts.sort_index()
    plt.figure(figsize=(12, 6))
    ax = monthly_stock_counts.plot(kind='bar')
    plt.title('Number of Stocks Included Each Month')
    plt.xlabel('Year-Month')
    plt.ylabel('Number of Stocks')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, fontsize=8)

    # Display every nth label on the x-axis (e.g., every 6th label)
    n = 6
    for index, label in enumerate(ax.get_xticklabels()):
        if index % n != 0:
            label.set_visible(False)

    plt.tight_layout()
    plt.savefig("2_IMG/MonthlyStocks.png")
    # plt.show()
    plt.pause(1)
    plt.close()



    ## Create a pivot table to make it easier to plot yearly trends
    pivot_table = monthly_stock_counts.unstack(level='year')
    ## Use a colormap with more than 22 distinct colors
    colors = plt.cm.tab20(range(len(pivot_table.columns)))

    ## Plot the yearly trend of monthly_stock_counts per month
    plt.figure(figsize=(12, 6))
    pivot_table.plot(marker='o', color=colors)
    plt.title('Yearly Trend of Monthly Stock Counts')
    plt.xlabel('Month')
    plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.ylabel('Number of Stocks')
    plt.grid(True)

    # Create a separate legend outside the plot area
    labels = [str(year) for year in pivot_table.columns]
    handles, _ = plt.gca().get_legend_handles_labels()
    plt.legend(handles, labels, title='Year', loc='upper left', bbox_to_anchor=(1, 1), prop={'size': 8})




    plt.tight_layout()
    plt.savefig("2_IMG/YearlyMonthlyStocks.png")
    # plt.show()
    plt.pause(1)
    plt.close()


    ## Output the pivot_table result to an Excel file
    pivot_table.to_excel('2_WIP/YearlyMonthlyStockCounts.xlsx', index=True)




def getDistribution_returnField(df,returns_column):

    ## Basic: Visualize the distribution of the returns field
    plt.figure(figsize=(8, 6))
    plt.hist(df[returns_column], bins=100, edgecolor='k', alpha=0.7)
    plt.title(f'Distribution of {returns_column}')
    plt.xlabel('Return')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(f'2_IMG/{returns_column}_Distribution.png')
    # plt.show()
    plt.pause(1)
    plt.close()


    ## use Using Z-Scores: to find outliers
    # Calculate z-scores for the return field
    z_scores = stats.zscore(df[returns_column])
    df[returns_column + '_zscore'] = z_scores

    # Create a figure with two subplots
    plt.figure(figsize=(12, 5))

    # Plot the original return field distribution
    plt.subplot(1, 2, 1)
    plt.hist(df[returns_column], bins=100, edgecolor='black', alpha=0.7)
    plt.title(f'Original {returns_column} Distribution')
    plt.xlabel('Return Field Values')
    plt.ylabel('Frequency')
    plt.grid(True)

    # Plot the z-score distribution
    plt.subplot(1, 2, 2)
    plt.hist(df[returns_column + '_zscore'], bins=100, edgecolor='black', color='orange')
    plt.title(f'Z-Scores for {returns_column}')
    plt.xlabel('Z-Scores')
    plt.ylabel('Frequency')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'2_IMG/{returns_column}_CombinedReturnFieldDistributions.png')
    # plt.show()
    plt.pause(1)
    plt.close()


    df[returns_column + '_zscore_copy']=df[returns_column + '_zscore'] #copy to a new column, ready to use

    ### Define the lower and upper percentiles (1st and 99th percentiles)
    lower_percentile = 1
    upper_percentile = 99

    ### Calculate the percentile values
    lower_limit = df[returns_column + '_zscore_copy'].quantile(lower_percentile / 100)
    upper_limit = df[returns_column + '_zscore_copy'].quantile(upper_percentile / 100)

    ### Winsorize the 'zscore_copy' column
    df[returns_column + '_zscore_winsorized'] = df[returns_column + '_zscore_copy'].clip(lower_limit, upper_limit)



    # Visualize the winsorized z-score distribution
    plt.figure(figsize=(8, 6))
    plt.hist(df[returns_column + '_zscore_winsorized'], bins=100, edgecolor='k', alpha=0.7, color='green')
    plt.title(f'Winsorized Z-Scores for {returns_column}')
    plt.xlabel('Winsorized Z-Scores')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(f'2_IMG/{returns_column}_ZScore_Winsorized.png')
    # plt.show()
    plt.pause(1)
    plt.close()

    # Find upside outliers (values above upper_limit)
    upside_outliers = df[df[returns_column + '_zscore_copy'] > upper_limit]
    # Find downside outliers (values below lower_limit)
    downside_outliers = df[df[returns_column + '_zscore_copy'] < lower_limit]

    # Print and save upside outliers
    print(f'Upside outliers for {returns_column}:')
    print(upside_outliers)
    upside_outliers.to_excel(f'2_WIP/{returns_column}_upside_outliers.xlsx', index=True)

    # Print and save downside outliers
    print(f'Downside outliers for {returns_column}:')
    print(downside_outliers)
    downside_outliers.to_excel(f'2_WIP/{returns_column}_downside_outliers.xlsx', index=True)



def getCoverage_returnField(df,returns_column):

    #### Pivot the DataFrame to check for missing return data by date and stock
    pivot_table = df.pivot_table(index='date', columns='stock_id', values=returns_column, aggfunc='count')

    #### Find dates where return data is missing for each stock
    missing_data = pivot_table.isnull()

    #### Print the pivot table with missing return data information
    print("Pivot Table: Missing Return Data (True means missing):")
    # print(missing_data)
    ## Output the pivot_table result to an Excel file
    missing_data.to_excel(f'2_WIP/{returns_column}_MissingReturnData(True means missing).xlsx', index=True)

    ## ---- Count the unique stocks that have missing data -----
    # Count the number of True values for each stock
    missing_data_counts = missing_data.sum()
    # Check for stocks with at least one True value (indicating missing data)
    stocks_with_missing_data = missing_data_counts[missing_data_counts > 0]
    # Count the unique stocks that have missing data
    unique_stock_count = len(stocks_with_missing_data)
    print(f"Total unique stocks with missing data for {returns_column}: {unique_stock_count}")

    ###


    ### Calculate % not missing for each stock
    data_coverage = (pivot_table.notnull().sum() / len(pivot_table)) * 100

    ### Filter stocks with return data
    stocks_with_data = data_coverage[data_coverage > 0]

    #### Define a threshold for low coverage
    low_coverage_threshold = 50
    #### Filter stocks with low coverage
    low_coverage_stocks = stocks_with_data[stocks_with_data < low_coverage_threshold]

    ####  Plot the coverage for stocks with return data using a bar chart
    plt.figure(figsize=(10, 6))
    bars = plt.bar(stocks_with_data.index, stocks_with_data.values)
    plt.title(f'Coverage of {returns_column} for Stocks with Data')
    plt.xlabel('Stock ID')
    plt.ylabel('Coverage (%)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)

    #### Highlight low coverage stocks on the chart
    for idx, coverage in low_coverage_stocks.items():
        bars[stocks_with_data.index.get_loc(idx)].set_color('red')
        plt.text(stocks_with_data.index.get_loc(idx), coverage + 1, f'Stock:{idx}: {coverage:.2f}', rotation=90,
                 ha='center')

    plt.tight_layout()
    plt.savefig(f'2_IMG/Coverage of {returns_column} for Stocks with Data.png')
    # plt.show()
    plt.pause(1)
    plt.close()



    ###----------------- other charts ---------------------
    ### Create a timeline chart to visualize missing data over time
    timeline_missing_data = pivot_table.applymap(lambda x: 1 if pd.isna(x) else 0)
    missing_data_timeline = timeline_missing_data.sum(axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(missing_data_timeline.index, missing_data_timeline.values, marker='.')
    plt.title(f'Missing Data Timeline for {returns_column}')
    plt.xlabel('Date')
    plt.ylabel('Missing Data Count')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(f'2_IMG/Missing Data Timeline for {returns_column}.png')
    # plt.show()
    plt.pause(1)
    plt.close()


    ### Create a line plot to visualize missing data patterns
    missing_data_patterns = pivot_table.isnull().sum() / len(pivot_table)

    plt.figure(figsize=(10, 6))
    plt.plot(missing_data_patterns.index, missing_data_patterns.values, marker='x')
    plt.title(f'Missing Data Patterns for {returns_column}')
    plt.xlabel('Stock ID')
    plt.ylabel('Percentage Missing Data')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)

    ## Identify the top 10 stock IDs with the highest missing data patterns
    top_missing_stocks = missing_data_patterns.nlargest(10)

    for stock_id, missing_percentage in top_missing_stocks.items():
        plt.text(stock_id, missing_percentage, f'Stock:{stock_id}: {missing_percentage:.2f}', rotation=90, ha='center')

    plt.tight_layout()
    plt.savefig(f'2_IMG/Missing Data Patterns for {returns_column}.png')
    # plt.show()
    plt.pause(1)
    plt.close()



    #### Create a heatmap to visualize the missing data patterns across stocks and time
    # Sort the index (date) of the pivot_table in ascending order
    pivot_table = pivot_table.sort_index()
    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot_table.isnull(), cmap="coolwarm", cbar=False)
    plt.title(f'Missing Data Heatmap for {returns_column}')
    plt.xlabel('Stock ID')
    plt.ylabel('Date')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(f'2_IMG/Missing Data Heatmap for {returns_column}.png')
    # plt.show()
    plt.pause(1)
    plt.close()


def getHighestFrequency_hist(x_values,binNumber):
    ### find the range of the highest frequency bin in the histogram.
    pb_counts, pb_bins, _ = plt.hist(x_values, bins=binNumber, edgecolor='green')
    ## Find the bin with the highest frequency
    highest_frequency_bin = np.argmax(pb_counts)
    ## Get the range (bin edges) of the highest frequency bin
    highest_frequency_range = (pb_bins[highest_frequency_bin], pb_bins[highest_frequency_bin + 1])
    ## Get the frequency of the highest frequency bin
    highest_frequency = pb_counts[highest_frequency_bin]
    print(f"Range of values with the highest frequency: {highest_frequency_range}")
    print(f"Frequency of the highest frequency bin: {highest_frequency}")

def getDistribution_PbField(df):

    pb_values = df['Pb'].values

    ### Fit the scaler on the data and transform it to get standardized values
    scaler = StandardScaler()
    standardized_pb = scaler.fit_transform(pb_values.reshape(-1, 1))

    ### Create a DataFrame with the standardized values
    df['Standardized_Pb'] = standardized_pb

    ###  combined plot
    plt.figure(figsize=(12, 6))

    ### Plot the original 'Pb' distribution
    plt.subplot(1, 2, 1)
    plt.hist(pb_values, bins=100, edgecolor='black')
    plt.title('Original Pb Distribution')
    plt.xlabel('Pb Value')
    plt.ylabel('Frequency')
    plt.grid(True)

    ### Plot the standardized 'Pb' distribution
    plt.subplot(1, 2, 2)
    plt.hist(standardized_pb, bins=100, edgecolor='black', color='orange')
    plt.title('Standardized Pb Distribution')
    plt.xlabel('Standardized Value (Z-Score)')
    plt.ylabel('Frequency')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('2_IMG/CombinedPbDistributions.png')
    # plt.show()
    plt.pause(1)
    plt.close()

    print("pb values findings:\n")
    getHighestFrequency_hist(pb_values,100)
    print("standardized pb values findings:\n")
    getHighestFrequency_hist(standardized_pb, 100)



    return df



if __name__ == '__main__':
    print ("---------------------start ---------------\n")



###----------!!!!----Please edit the raw data path and choose correct file format----------------------
    raw_csv = "1_Rawdata/p2_class_project_data.csv"
    df_raw = readCSV(raw_csv)
    # print(df_raw)
    df=df_raw
    df=prepareData(df)


    # ### N1: Plot the number of stocks included in the data for each month of the provided history.
    getPlot_MonthlyStocks(df)



    ## N2: Visualize the distribution of the provided returns field.
    returns_column = 'R1M_Usd'
    getDistribution_returnField(df, returns_column)
    returns_column = 'Vol3Y_Usd'
    getDistribution_returnField(df, returns_column)
    returns_column = 'Mkt_Cap_3M_Usd'
    getDistribution_returnField(df, returns_column)

    #df.to_csv('winsorized_data.csv', index=False)




    #### N3:Plot the coverage (% not missing) of the return data field each period.
    returns_column = 'R1M_Usd'
    getCoverage_returnField(df,returns_column)
    returns_column = 'Vol3Y_Usd'
    getCoverage_returnField(df, returns_column)
    returns_column = 'Mkt_Cap_3M_Usd'
    getCoverage_returnField(df, returns_column)



    ##### N4: Visualize the distribution of the provided standardized price to book field.
    df=getDistribution_PbField(df)  ##added column: Standardized_Pb

    ### clean up dataframe, before export to csv
    columns_to_drop = [col for col in df.columns if "_zscore_copy" in col]
    df = df.drop(columns=columns_to_drop)
    # df.to_excel('2_WIP/df_final.xlsx', index=True)
    df.to_csv('2_WIP/df_final.csv', index=False)



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






    exit()





