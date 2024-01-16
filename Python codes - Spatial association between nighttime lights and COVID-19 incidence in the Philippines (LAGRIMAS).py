#!/usr/bin/env python
# coding: utf-8

# #  **Spatial association between nighttime lights and COVID-19 incidence in the Philippines**

# ### Ajay L. Lagrimas  
# BS Physics  
# National Institute of Physics  
# University of the Philippines  
# Diliman, Quezon City  

# ## Preliminary codes to set up packages

# In[ ]:


# Setting up the necessary packages for Google Earth Engine

try:
    import geemap, ee
except ModuleNotFoundError:
    if 'google.colab' in str(get_ipython()):
        print("Package not found, installing with pip in Google Colab...")
        get_ipython().system('pip install geemap')
    else:
        print("Package not found, installing with conda...")
        get_ipython().system('conda install -c conda-forge geemap -y')
    import geemap, ee


# In[ ]:


# Setting up other the necessary packages

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
from scipy.stats import spearmanr
import seaborn as sns
import ee
import matplotlib as mpl
from matplotlib.transforms import Bbox
from brokenaxes import brokenaxes

plt.rcParams['figure.dpi']=100


# In[ ]:


# Authenticating and initializing Google Earth

try:
        ee.Initialize()
except Exception as e:
        ee.Authenticate()
        ee.Initialize()


# In[ ]:


# Loading the FAO/GAUL/2015/level2 dataset for the Philippines

gaul_ph = ee.FeatureCollection("FAO/GAUL/2015/level2").filter(ee.Filter.eq('ADM0_NAME', 'Philippines'))

# Extracting the province names
province_list = gaul_ph.aggregate_array('ADM2_NAME').getInfo()
print(province_list)


# In[ ]:


# Creating a dataframe for Provincial Average SOL

provinces = ['Guimaras', 'Iloilo', 'Biliran', 'Leyte', 'Saranggani', 'South Cotabato', 'Compostela', 'Davao del Norte', 'Zamboanga Sibugay', 'Zamboanga Del Sur', 'Dinagat', 'Surigao Del Norte', 'Maguindanao', 'Shariff Kabunsuan', 'Abra', 'Apayao', 'Benguet', 'Ifugao', 'Kalinga', 'Mountain Province', 'Metropolitan Manila', 'Ilocos Norte', 'Ilocos Sur', 'La Union', 'Pangasinan', 'Batanes', 'Cagayan', 'Isabela', 'Nueva Vizcaya', 'Quirino', 'Albay', 'Camarines Norte', 'Camarines Sur', 'Catanduanes', 'Masbate', 'Sorsogon', 'Aklan', 'Antique', 'Capiz', 'Negros Occidental', 'Bohol', 'Cebu', 'Negros Oriental', 'Siquijor', 'Eastern Samar', 'Northern Samar', 'Southern Leyte', 'Samar', 'Agusan Del Norte', 'Agusan Del Sur', 'Surigao Del Sur', 'Lanao Del Sur', 'Sulu', 'Tawi-tawi', 'Basilan', 'Zamboanga Del Norte', 'Bukidnon', 'Camiguin', 'Misamis Occidental', 'Misamis Oriental', 'Lanao Del Norte', 'Davao Del Sur', 'Davao Oriental', 'North Cotabato', 'Sultan Kudarat', 'Bataan', 'Bulacan', 'Nueva Ecija', 'Pampanga', 'Tarlac', 'Zambales', 'Aurora', 'Batangas', 'Cavite', 'Laguna', 'Quezon', 'Rizal', 'Marinduque', 'Mindoro Occidental', 'Mindoro Oriental', 'Palawan', 'Romblon']

average_SOL_list = []

for province in provinces:
    result_df = pd.DataFrame(columns=['Province', 'Average_NTL'])

# Setting provincial boundary using FAO GAUL overlays
    viirs = ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG").filterDate('2019-01-01','2019-06-30') #Filter date is the timespan covered by NTL
    prov = ee.Feature(ee.FeatureCollection("FAO/GAUL/2015/level2").filter(ee.Filter.eq('ADM2_NAME', province)).first()).geometry() #Level2 in Gaul has a provincial resolution.
    provaoi = ee.Feature(ee.FeatureCollection("FAO/GAUL/2015/level2").filter(ee.Filter.eq('ADM2_NAME', province)))

# Reducer function will get the SOL
    def get_prov_sol(img):
        sol = img.reduceRegion(reducer=ee.Reducer.sum(), geometry=prov, scale=500, maxPixels=1e9).get('avg_rad')
        return img.set('date', img.date().format()).set('SOL',sol)

    prov_sol = viirs.map(get_prov_sol)
    nested_list = prov_sol.reduceColumns(ee.Reducer.toList(2), ['date','SOL']).values().get(0)

# Converting the dataframe
    soldf = pd.DataFrame(nested_list.getInfo(), columns=['date','SOL'])
    soldf['date'] = pd.to_datetime(soldf['date'])
    soldf = soldf.set_index('date')
    soldf

# Calculating the average of the 'SOL' column
    average_SOL = soldf['SOL'].mean()
    average_SOL_list.append([province, average_SOL])

# Create a DataFrame from the accumulated data
provsol_df = pd.DataFrame(average_SOL_list, columns=['Province', 'Average_SOL'])

# Arrange in alphabetical order.
sorted_provsol_df = provsol_df.sort_values(by='Province')

# Print the sorted DataFrame
sorted_provsol_df


# In[ ]:


# Mounting Google Drive to import DOH Data Drop

from google.colab import drive
drive.mount("/content/drive")


# In[ ]:


# OLD DOH DATA DROP (Retrieved: May 2023)

file_path2 = "/content/drive/MyDrive/COVID-19 Cases DOH Data Drop" # Replace with Google Drive folder containing Data Drop

# Replace with the correct location of the DOH Data Drop files
batch0 = pd.read_csv("/content/drive/MyDrive/COVID-19 Cases DOH Data Drop/DOH COVID Data Drop_ 20230422 - 04 Case Information_batch_0.csv")
batch1 = pd.read_csv("/content/drive/MyDrive/COVID-19 Cases DOH Data Drop/DOH COVID Data Drop_ 20230422 - 04 Case Information_batch_1.csv")
batch2 = pd.read_csv("/content/drive/MyDrive/COVID-19 Cases DOH Data Drop/DOH COVID Data Drop_ 20230422 - 04 Case Information_batch_2.csv")
batch3 = pd.read_csv("/content/drive/MyDrive/COVID-19 Cases DOH Data Drop/DOH COVID Data Drop_ 20230422 - 04 Case Information_batch_3.csv")
batch4 = pd.read_csv("/content/drive/MyDrive/COVID-19 Cases DOH Data Drop/DOH COVID Data Drop_ 20230422 - 04 Case Information_batch_4.csv")


# In[ ]:


# Extracting data from csv files

all_files = glob.glob(file_path2 + "/*.csv")

li = []

for filename in all_files:
  df = pd.read_csv(filename, index_col=None, header=0)
  li.append(df)

frame = pd.concat(li, axis=0, ignore_index=True)


# ## Generating scatter plots for correlation analysis

# In[ ]:


# Creating a dataframe for the number of recorded COVID-19 cases per province

# Define the date range. Replace with the desired time span.
start_date = '2020-01-01'
end_date = '2023-04-30'

# Convert the column to datetime (if not yet in datetime)
frame['DateRepConf'] = pd.to_datetime(frame['DateRepConf'])  

# Filter the DataFrame to include only rows within the specified date range
batch = frame[(frame['DateRepConf'] >= start_date) & (frame['DateRepConf'] <= end_date)]

batch = batch.sort_values(by='ProvRes', ascending = True, inplace = False, kind = 'quicksort', na_position = 'last')
newbatch = batch['ProvRes'].value_counts(ascending = True)
newbatch_df = newbatch.reset_index()
newbatch_df

# Arrange provinces in alphabetical order
sorted_newbatch_df = newbatch_df.sort_values(by='index')
sorted_newbatch_df


# In[ ]:


# Importing a csv file for the manually matched provincial DOH Data Drop and mean SOL

file_path_correlold = "/content/drive/MyDrive/Data Correlation/Tri-Monthly COVID vs NTL - Sheet1.csv"
# The file being imported in this code is a csv file where the number of cases and mean SOL for each province are given

old_df = pd.read_csv(file_path_correlold)
old_df


# In[ ]:


# Generating a scatter plot for the trimonthly mean SOL and number of COVID-19 cases

x_colname = 'MEAN SOL'
y_colname = 'NEW CASES'

plt.figure(figsize=(8, 6))

# Filter and plot points where 'PROVINCE' is not 'NCR' in orange
filtered_df = old_df[old_df['PROVINCE'] != 'NCR']
plt.scatter(filtered_df[x_colname], filtered_df[y_colname], color='orange', label='Other Provinces')

# Filter and plot points where 'PROVINCE' is 'NCR' in green
ncr_df = old_df[old_df['PROVINCE'] == 'NCR']
plt.scatter(ncr_df[x_colname], ncr_df[y_colname], color='green', label='NCR')

# Calculate the Spearman's rank correlation coefficient for all points
correlation_all, _ = spearmanr(old_df[x_colname], old_df[y_colname])
plt.annotate(f'Spearman = {correlation_all:.2f}', xy=(0.7, 0.9), xycoords='axes fraction', fontsize=10)

plt.xlabel('Mean SOL')
plt.ylabel('Recorded COVID-19 cases')
plt.title('Trimonthly mean sum of lights and COVID-19 cases for all provinces')
plt.legend()
plt.tight_layout()
plt.show()


# In[ ]:


# Generating a scatter plot for the trimonthly mean SOL and COVID-19 deaths

file_path_mortality = "/content/drive/MyDrive/Data Correlation/Mortality Rates vs NTL - Sheet1.csv"
mort_df = pd.read_csv(file_path_mortality)

# Calculate the "MORTALITY RATE" and add it as a new column
mort_df['MORTALITY RATE'] = (mort_df['DEATHS'] / mort_df['COVID CASES']) * 100
#Clean data by removing elements with NaN
mort_df_cleaned = mort_df.dropna(subset=['MORTALITY RATE'])

# Create a scatterplot with points colored orange for all provinces
plt.figure(figsize=(8, 6))
plt.scatter(mort_df_cleaned['MEAN SOL'], mort_df_cleaned['DEATHS'], color='brown', alpha=0.8, label='Other Provinces')

# Filter the data for points with "NCR" as the province and color them green
ncr_df_mort = mort_df_cleaned[mort_df_cleaned['PROVINCE'] == 'NCR']
plt.scatter(ncr_df_mort['MEAN SOL'], ncr_df_mort['DEATHS'], color='orange', alpha=0.8, label='NCR')

plt.xlabel('Mean SOL')
plt.ylabel('Recorded COVID-related deaths')
plt.title('Trimonthly mean sum of lights and COVID-related deaths for all provinces')

# Calculate the Spearman correlation
correlation, _ = spearmanr(mort_df_cleaned['MEAN SOL'], mort_df_cleaned['DEATHS'])
plt.text(0.1, 0.9, f'Spearman = {correlation:.2f}', transform=plt.gca().transAxes, fontsize=10)

plt.legend()
plt.tight_layout()
plt.show()


# In[ ]:


# Importing a csv file for the manually matched provincial COVID-19 cases and mean SOL with provincial population count

file_path_incidence = "/content/drive/MyDrive/Data Correlation/Incidence Rates vs NTL - Sheet1.csv"
# The file being imported in this code is a csv file where the number of cases, mean SOL, and population for each province are given

inc_df = pd.read_csv(file_path_incidence)
inc_df


# In[ ]:


# Generating a scatter plot for the trimonthly mean SOL and COVID-19 incidence rate

# Calculate the "INCIDENCE RATE" and add it as a new column
inc_df['INCIDENCE RATE'] = (inc_df['NEW CASES'] / inc_df['POPULATION COUNT']) * 100

# Create a scatterplot with points colored orange for all provinces
plt.figure(figsize=(8, 6))
plt.scatter(inc_df['MEAN SOL'], inc_df['INCIDENCE RATE'], color='coral', alpha=0.8, label='Other Provinces')

# Filter the data for points with "NCR" as the province and color them green
ncr_df = inc_df[inc_df['PROVINCE'] == 'NCR']
plt.scatter(ncr_df['MEAN SOL'], ncr_df['INCIDENCE RATE'], color='purple', alpha=0.8, label='NCR')

plt.xlabel('Mean SOL')
plt.ylabel('COVID-19 incidence rate')
plt.title('Trimonthly mean sum of lights and COVID-19 incidence rate for all provinces')

# Calculate the Spearman correlation
correlation, _ = spearmanr(inc_df['MEAN SOL'], inc_df['INCIDENCE RATE'])
plt.text(0.1, 0.9, f'Spearman = {correlation:.2f}', transform=plt.gca().transAxes, fontsize=10)

plt.legend()
plt.tight_layout()
plt.show()


# In[ ]:


# Generating a scatter plot for the trimonthly mean SOL and COVID-19 mortality rate

file_path_mortality = "/content/drive/MyDrive/Data Correlation/Mortality Rates vs NTL - Sheet1.csv"
mort_df = pd.read_csv(file_path_mortality)

# Calculate the "MORTALITY RATE" and add it as a new column
mort_df['MORTALITY RATE'] = (mort_df['DEATHS'] / mort_df['COVID CASES']) * 100
#Clean data by removing elements with NaN
mort_df_cleaned = mort_df.dropna(subset=['MORTALITY RATE'])

# Create a scatterplot with points colored orange for all provinces
plt.figure(figsize=(8, 6))
plt.scatter(mort_df_cleaned['MEAN SOL'], mort_df_cleaned['MORTALITY RATE'], color='gray', alpha=0.8, label='Other Provinces')

# Filter the data for points with "NCR" as the province and color them green
ncr_df_mort = mort_df_cleaned[mort_df_cleaned['PROVINCE'] == 'NCR']
plt.scatter(ncr_df_mort['MEAN SOL'], ncr_df_mort['MORTALITY RATE'], color='black', alpha=0.8, label='NCR')

plt.xlabel('Mean SOL')
plt.ylabel('COVID-19 mortality rate')
plt.title('Trimonthly mean sum of lights and mortality rate for all provinces')

# Calculate the Spearman correlation
correlation, _ = spearmanr(mort_df_cleaned['MEAN SOL'], mort_df_cleaned['MORTALITY RATE'])
plt.text(0.1, 0.9, f'Spearman = {correlation:.2f}', transform=plt.gca().transAxes, fontsize=10)

plt.legend()
plt.tight_layout()
plt.show()


# In[ ]:


# Generating a scatter plot for the mean SOL and cases for the entire course of the pandemic

x_colname = 'MEAN SOL'
y_colname = 'TOTAL CASES'

file_path_entire = "/content/drive/MyDrive/Data Correlation/ENTIRE COURSE NTL vs COVID - Copy of Sheet1.csv"
cases_df = pd.read_csv(file_path_entire)

plt.figure(figsize=(8, 6))

# Filter and plot points where 'PROVINCE' is not 'NCR' in orange
filtered_df = cases_df[cases_df['PROVINCE'] != 'NCR']
plt.scatter(filtered_df[x_colname], filtered_df[y_colname], color='red', label='Other Provinces')

# Filter and plot points where 'PROVINCE' is 'NCR' in green
ncr_df = cases_df[cases_df['PROVINCE'] == 'NCR']
plt.scatter(ncr_df[x_colname], ncr_df[y_colname], color='orange', label='NCR')

# Calculate the Spearman rank correlation coefficient for all points
correlation_all, _ = spearmanr(cases_df[x_colname], cases_df[y_colname])
plt.annotate(f'Spearman = {correlation_all:.2f}', xy=(0.7, 0.9), xycoords='axes fraction', fontsize=10)

plt.xlabel('Mean SOL')
plt.ylabel('Total COVID-19 Cases')
plt.title('Overall mean sum of lights and COVID-19 cases (Jan 2020 to April 2023)')
plt.legend()
plt.tight_layout()
plt.show()


# In[ ]:


# Generating a scatter plot for the mean SOL and incidence rate for the entire course of the pandemic

x_colname = 'MEAN SOL'
y_colname = 'INFECTION RATE'

file_path_entire = "/content/drive/MyDrive/Data Correlation/ENTIRE COURSE NTL vs COVID - Copy of Sheet1.csv"
cases_df = pd.read_csv(file_path_entire)

plt.figure(figsize=(8, 6))

# Filter and plot points where 'PROVINCE' is not 'NCR' in orange
filtered_df = cases_df[cases_df['PROVINCE'] != 'NCR']
plt.scatter(filtered_df[x_colname], filtered_df[y_colname], color='coral', label='Other Provinces')

# Filter and plot points where 'PROVINCE' is 'NCR' in green
ncr_df = cases_df[cases_df['PROVINCE'] == 'NCR']
plt.scatter(ncr_df[x_colname], ncr_df[y_colname], color='purple', label='NCR')

# Calculate the Spearman rank correlation coefficient for all points
correlation_all, _ = spearmanr(cases_df[x_colname], cases_df[y_colname])
plt.annotate(f'Spearman = {correlation_all:.2f}', xy=(0.7, 0.9), xycoords='axes fraction', fontsize=10)

plt.xlabel('Mean SOL')
plt.ylabel('COVID-19 incidence rate')
plt.title('Overall mean sum of lights and incidence rate (1 Jan 2020 to 21 April 2023)')
plt.legend()
plt.tight_layout()
plt.show()


# In[ ]:


# Generating a scatter plot for the mean SOL and deaths for the entire course of the pandemic

x_colname = 'MEAN SOL'
y_colname = 'TOTAL DEATHS'

file_path_entire = "/content/drive/MyDrive/Data Correlation/ENTIRE COURSE NTL vs COVID - Copy of Sheet1.csv"
cases_df = pd.read_csv(file_path_entire)

plt.figure(figsize=(8, 6))

# Filter and plot points where 'PROVINCE' is not 'NCR' in orange
filtered_df = cases_df[cases_df['PROVINCE'] != 'NCR']
plt.scatter(filtered_df[x_colname], filtered_df[y_colname], color='brown', label='Other Provinces')

# Filter and plot points where 'PROVINCE' is 'NCR' in green
ncr_df = cases_df[cases_df['PROVINCE'] == 'NCR']
plt.scatter(ncr_df[x_colname], ncr_df[y_colname], color='orange', label='NCR')

# Calculate the Spearman rank correlation coefficient for all points
correlation_all, _ = spearmanr(cases_df[x_colname], cases_df[y_colname])
plt.annotate(f'Spearman = {correlation_all:.2f}', xy=(0.7, 0.8), xycoords='axes fraction', fontsize=10)

plt.xlabel('Mean SOL')
plt.ylabel('Total COVID-related deaths')
plt.title('Overall mean sum of lights and COVID-related deaths (1 Jan 2020 to 21 April 2023)')
plt.legend()
plt.tight_layout()
plt.show()


# In[ ]:


# Generating a scatter plot for the mean SOL and mortality rate for the entire course of the pandemic

x_colname = 'MEAN SOL'
y_colname = 'MORTALITY RATE'

file_path_entire = "/content/drive/MyDrive/Data Correlation/ENTIRE COURSE NTL vs COVID - Copy of Sheet1.csv"
cases_df = pd.read_csv(file_path_entire)

plt.figure(figsize=(8, 6))

# Filter and plot points where 'PROVINCE' is not 'NCR' in orange
filtered_df = cases_df[cases_df['PROVINCE'] != 'NCR']
plt.scatter(filtered_df[x_colname], filtered_df[y_colname], color='gray', label='Other Provinces')

# Filter and plot points where 'PROVINCE' is 'NCR' in green
ncr_df = cases_df[cases_df['PROVINCE'] == 'NCR']
plt.scatter(ncr_df[x_colname], ncr_df[y_colname], color='black', label='NCR')

# Calculate the Spearman rank correlation coefficient for all points
correlation_all, _ = spearmanr(cases_df[x_colname], cases_df[y_colname])
plt.annotate(f'Spearman = {correlation_all:.2f}', xy=(0.7, 0.8), xycoords='axes fraction', fontsize=10)

plt.xlabel('Mean SOL')
plt.ylabel('COVID-19 mortality rate')
plt.title('Overall mean sum of lights and mortality rate (1 Jan 2020 to 21 April 2023)')
plt.legend()
plt.tight_layout()
plt.show()


# In[ ]:


# Generating scatter plot for monthly NTL deviation and COVID-19 cases

ntlsol_dev = "/content/drive/MyDrive/Data Correlation/Deviation vs NTL - Deviation vs NTL.csv"
ntlsol_dev_df = pd.read_csv(ntlsol_dev)

# Create a scatterplot with points colored orange for all provinces
plt.figure(figsize=(8, 6))
plt.scatter(ntlsol_dev_df['Cases'], ntlsol_dev_df['Deviation'], color='coral', alpha=0.8, label='Other Provinces')

# Filter the data for points with "NCR" as the province and color them green
ncr_df = ntlsol_dev_df[ntlsol_dev_df['PROVINCE'] == 'NCR']
plt.scatter(ncr_df['Cases'], ncr_df['Deviation'], color='purple', alpha=0.8, label='NCR')
plt.axhline(y=0, color='black', linestyle='-', linewidth=1)

plt.xlabel('Monthly COVID-19 cases')
plt.ylabel('Monthly SOL deviation from average')
plt.title('Monthly deviation of NTL from average and COVID-19 cases per province')

# Calculate the Spearman correlation
correlation, _ = spearmanr(ntlsol_dev_df['Cases'], ntlsol_dev_df['Deviation'])
plt.text(0.7, 0.7, f'Spearman = {correlation:.2f}', transform=plt.gca().transAxes, fontsize=10)

plt.legend()
plt.tight_layout()
plt.show()


# In[ ]:


# Generating a scatter plot for monthly NTL deviation and COVID-19 cases, and color-coding points based on outbreak dates

ntlsol_dev = "/content/drive/MyDrive/Data Correlation/Deviation vs NTL - Deviation vs NTL.csv"
ntlsol_dev_df = pd.read_csv(ntlsol_dev)

plt.figure(figsize=(8, 6))

# Define the multiple highlight dates
highlight_dates = ['5/31/20', '6/30/20', '7/31/20','8/31/20','9/30/20','10/31/20','11/30/20','12/31/20','1/31/21']

for date in highlight_dates:
    mask = ntlsol_dev_df['Date'] == date
    plt.scatter(
        ntlsol_dev_df[~mask]['Cases'], ntlsol_dev_df[~mask]['Deviation'],
        color='white', alpha=0
    )
    plt.scatter(
        ntlsol_dev_df[mask]['Cases'], ntlsol_dev_df[mask]['Deviation'],
        alpha=0.5, color='red'
    )

plt.axhline(y=0, color='black', linestyle='-', linewidth=1)

plt.xlabel('Monthly COVID-19 cases')
plt.ylabel('Monthly SOL deviation from average')
plt.title('1st Outbreak monthly NTL deviation and COVID-19 cases')

# Calculate the Spearman correlation
correlation, _ = spearmanr(ntlsol_dev_df['Cases'], ntlsol_dev_df['Deviation'])
plt.text(0.7, 0.7, f'Spearman = {correlation:.2f}', transform=plt.gca().transAxes, fontsize=10)

plt.legend()
plt.tight_layout()
plt.show()


# In[ ]:


# Spearman's rank coefficient computation and p-value

inc_df['INCIDENCE RATE'] = (inc_df['NEW CASES'] / inc_df['POPULATION COUNT']) * 100
spearman_corr, p_value = spearmanr(inc_df['INCIDENCE RATE'], inc_df['MEAN SOL'])

print(f"Spearman Rank Correlation: {spearman_corr}")
print(f"P-Value: {p_value}")


# ## Generating time series plots

# In[ ]:


# Generating a time series for the monthly mean SOL for all the provinces

# Retrieiving the NTL data for the provinces
provinces = ['Guimaras', 'Iloilo', 'Biliran', 'Leyte', 'Saranggani', 'South Cotabato', 'Compostela', 'Davao del Norte', 'Zamboanga Sibugay', 'Zamboanga Del Sur', 'Dinagat', 'Surigao Del Norte', 'Maguindanao', 'Shariff Kabunsuan', 'Abra', 'Apayao', 'Benguet', 'Ifugao', 'Kalinga', 'Mountain Province', 'Metropolitan Manila', 'Ilocos Norte', 'Ilocos Sur', 'La Union', 'Pangasinan', 'Batanes', 'Cagayan', 'Isabela', 'Nueva Vizcaya', 'Quirino', 'Albay', 'Camarines Norte', 'Camarines Sur', 'Catanduanes', 'Masbate', 'Sorsogon', 'Aklan', 'Antique', 'Capiz', 'Negros Occidental', 'Bohol', 'Cebu', 'Negros Oriental', 'Siquijor', 'Eastern Samar', 'Northern Samar', 'Southern Leyte', 'Samar', 'Agusan Del Norte', 'Agusan Del Sur', 'Surigao Del Sur', 'Lanao Del Sur', 'Sulu', 'Tawi-tawi', 'Basilan', 'Zamboanga Del Norte', 'Bukidnon', 'Camiguin', 'Misamis Occidental', 'Misamis Oriental', 'Lanao Del Norte', 'Davao Del Sur', 'Davao Oriental', 'North Cotabato', 'Sultan Kudarat', 'Bataan', 'Bulacan', 'Nueva Ecija', 'Pampanga', 'Tarlac', 'Zambales', 'Aurora', 'Batangas', 'Cavite', 'Laguna', 'Quezon', 'Rizal', 'Marinduque', 'Mindoro Occidental', 'Mindoro Oriental', 'Palawan', 'Romblon']
def get_province_sol(province):
    viirs = ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG").filterDate('2020-03-08', '2023-04-21')
    prov = ee.Feature(ee.FeatureCollection("FAO/GAUL/2015/level2").filter(ee.Filter.eq('ADM2_NAME', province)).first()).geometry()

    def get_prov_sol(img):
        sol = img.reduceRegion(reducer=ee.Reducer.sum(), geometry=prov, scale=500, maxPixels=1e9).get('avg_rad')
        return img.set('date', img.date().format()).set('SOL', sol)

    prov_sol = viirs.map(get_prov_sol)
    nested_list = prov_sol.reduceColumns(ee.Reducer.toList(2), ['date', 'SOL']).values().get(0)

    soldf = pd.DataFrame(nested_list.getInfo(), columns=['date', 'SOL'])
    soldf['date'] = pd.to_datetime(soldf['date'])
    soldf = soldf.set_index('date')

    return soldf

# Create a dictionary to store the SOL data for each province
province_data = {}
for province in provinces:
    province_data[province] = get_province_sol(province)

# Plot all NTL series on the same plot and creating a legend for Metro Manila
fig, ax = plt.subplots(figsize=(12, 6))
for province, data in province_data.items():
    if province == "Metropolitan Manila":
        sns.lineplot(data=data['SOL'], label=province, ax=ax)
    else:
        sns.lineplot(data=data['SOL'], ax=ax)

ax.set_ylabel('Mean SOL', fontsize=10)
ax.set_xlabel('Date', fontsize=10)
ax.set_title('Monthly mean sum of lights (SOL) for all provinces in the Philippines (Jan 2020 to April 2023)')
ax.legend()
plt.show()


# In[ ]:


# Generating a time series plot for the daily COVID-19 cases nationwide

frame['DateRepConf'] = pd.to_datetime(frame['DateRepConf'])  # Convert the column to datetime if it's not already

# Sort the DataFrame by 'DateRepConf'
frame = frame.sort_values(by='DateRepConf')

# Group by date and count the number of cases for each date
daily_cases = frame.groupby('DateRepConf').size()

# Plot the number of cases per day
plt.figure(figsize=(12, 6))
plt.plot(daily_cases.index, daily_cases.values, linestyle='-', color='red')
plt.xlabel('Date')
plt.ylabel('Recorded COVID-19 Cases')
plt.title('Nationwide monthly COVID-19 cases (Jan 2020 to April 2023)')
plt.xticks(rotation=0, size = 8)
plt.show()


# In[ ]:


# Generating a time series of the nationwide daily COVID-19 cases with highlighted time span for outbreaks

frame['DateRepConf'] = pd.to_datetime(frame['DateRepConf'])  # Convert the column to datetime if it's not already
frame = frame.sort_values(by='DateRepConf')
daily_cases = frame.groupby('DateRepConf').size()

# Define the start and end dates of the selected time frame
selected_start_date = '2020-05-30'
selected_end_date = '2021-01-01'

selected_start_date2 = '2021-03-01'
selected_end_date2 = '2021-07-25'

selected_start_date3 = '2021-07-25'
selected_end_date3 = '2021-12-25'

selected_start_date4 = '2021-12-25'
selected_end_date4 = '2022-04-01'

selected_start_date5 = '2022-07-01'
selected_end_date5 = '2023-02-01'

# Plot the number of cases per day
plt.figure(figsize=(12, 6))
plt.plot(daily_cases.index, daily_cases.values, linestyle='-', color='red')
plt.xlabel('Date')
plt.ylabel('Recorded COVID-19 Cases')
plt.title('Nationwide monthly COVID-19 cases and observed peaks')
plt.xticks(rotation=0, size=8)

# Highlight the selected time frame
plt.axvspan(selected_start_date, selected_end_date, color='lightcoral', alpha=0.5)
plt.axvspan(selected_start_date2, selected_end_date2, color='lightgreen', alpha=0.5)
plt.axvspan(selected_start_date3, selected_end_date3, color='lightblue', alpha=0.5)
plt.axvspan(selected_start_date4, selected_end_date4, color='yellow', alpha=0.5)
plt.axvspan(selected_start_date5, selected_end_date5, color='lightpink', alpha=0.5)

plt.show()


# In[ ]:


# Generating a time series plot for the nationwide daily COVID-19 cases with highlighted time span for COVID-19 Variants

# Defining the time span of detection of each outbreak based on DOH Data
selected_start_date = '2021-01-21'  # Alpha
selected_end_date = '2021-01-21'

selected_start_date2 = '2021-02-12'  # Theta
selected_end_date2 = '2021-02-12'

selected_start_date3 = '2021-02-28'  # Beta
selected_end_date3 = '2021-02-28'

selected_start_date4 = '2021-03-12'  # Gamma
selected_end_date4 = '2021-03-12'

selected_start_date5 = '2021-05-09'  # Delta
selected_end_date5 = '2021-05-09'

selected_start_date6 = '2021-08-15'  # Lambda
selected_end_date6 = '2021-08-15'

selected_start_date7 = '2021-12-15'  # Omicron
selected_end_date7 = '2021-12-15'

highlighted_spans = [
    (selected_start_date, selected_end_date, 'Alpha Variant', 'orange'),
    (selected_start_date2, selected_end_date2, 'Theta Variant', 'yellow'),
    (selected_start_date3, selected_end_date3, 'Beta Variant', 'green'),
    (selected_start_date4, selected_end_date4, 'Gamma Variant', 'turquoise'),
    (selected_start_date5, selected_end_date5, 'Delta Variant', 'blue'),
    (selected_start_date6, selected_end_date6, 'Lambda Variant', 'violet'),
    (selected_start_date7, selected_end_date7, 'Omicron Variant', 'purple')
]


frame['DateRepConf'] = pd.to_datetime(frame['DateRepConf'])  # Convert the column to datetime if it's not already
frame = frame.sort_values(by='DateRepConf')
daily_cases = frame.groupby('DateRepConf').size()


fig, ax = plt.subplots(figsize=(12, 6))
plt.plot(daily_cases.index, daily_cases.values, linestyle='-', color='red')
plt.xlabel('Date')
plt.ylabel('Recorded COVID-19 Cases')
plt.title('Nationwide monthly COVID-19 cases and variants detection')
plt.xticks(rotation=0, size=8)

for start_date, end_date, label_text, color in highlighted_spans:
    ax.axvspan(start_date, end_date, color=color, alpha=0.5)

legend_patches = [plt.Rectangle((0, 0), 1, 1, color=color) for _, _, _, color in highlighted_spans]
legend_labels = [label_text for _, _, label_text, _ in highlighted_spans]

ax.legend(legend_patches, legend_labels, loc="upper left")
plt.tight_layout()
plt.show()


# In[ ]:


# Generating a time series plot superimposing the monthly mean SOL and COVID-19 outbreaks

# Retrieving the NTL data per province
provinces = ['Guimaras', 'Iloilo', 'Biliran', 'Leyte', 'Saranggani', 'South Cotabato', 'Compostela', 'Davao del Norte', 'Zamboanga Sibugay', 'Zamboanga Del Sur', 'Dinagat', 'Surigao Del Norte', 'Maguindanao', 'Shariff Kabunsuan', 'Abra', 'Apayao', 'Benguet', 'Ifugao', 'Kalinga', 'Mountain Province', 'Metropolitan Manila', 'Ilocos Norte', 'Ilocos Sur', 'La Union', 'Pangasinan', 'Batanes', 'Cagayan', 'Isabela', 'Nueva Vizcaya', 'Quirino', 'Albay', 'Camarines Norte', 'Camarines Sur', 'Catanduanes', 'Masbate', 'Sorsogon', 'Aklan', 'Antique', 'Capiz', 'Negros Occidental', 'Bohol', 'Cebu', 'Negros Oriental', 'Siquijor', 'Eastern Samar', 'Northern Samar', 'Southern Leyte', 'Samar', 'Agusan Del Norte', 'Agusan Del Sur', 'Surigao Del Sur', 'Lanao Del Sur', 'Sulu', 'Tawi-tawi', 'Basilan', 'Zamboanga Del Norte', 'Bukidnon', 'Camiguin', 'Misamis Occidental', 'Misamis Oriental', 'Lanao Del Norte', 'Davao Del Sur', 'Davao Oriental', 'North Cotabato', 'Sultan Kudarat', 'Bataan', 'Bulacan', 'Nueva Ecija', 'Pampanga', 'Tarlac', 'Zambales', 'Aurora', 'Batangas', 'Cavite', 'Laguna', 'Quezon', 'Rizal', 'Marinduque', 'Mindoro Occidental', 'Mindoro Oriental', 'Palawan', 'Romblon']
def get_province_sol(province):
    viirs = ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG").filterDate('2020-03-08', '2023-04-21')
    prov = ee.Feature(ee.FeatureCollection("FAO/GAUL/2015/level2").filter(ee.Filter.eq('ADM2_NAME', province)).first()).geometry()

    def get_prov_sol(img):
        sol = img.reduceRegion(reducer=ee.Reducer.sum(), geometry=prov, scale=500, maxPixels=1e9).get('avg_rad')
        return img.set('date', img.date().format()).set('SOL', sol)

    prov_sol = viirs.map(get_prov_sol)
    nested_list = prov_sol.reduceColumns(ee.Reducer.toList(2), ['date', 'SOL']).values().get(0)

    soldf = pd.DataFrame(nested_list.getInfo(), columns=['date', 'SOL'])
    soldf['date'] = pd.to_datetime(soldf['date'])
    soldf = soldf.set_index('date')

    return soldf

# Create a dictionary to store the SOL data for each province
province_data = {}
for province in provinces:
    province_data[province] = get_province_sol(province)

# Plot all NTL series on the same plot
fig, ax = plt.subplots(figsize=(12, 6))
for province, data in province_data.items():
    if province == "Metropolitan Manila":
        sns.lineplot(data=data['SOL'], label=province, ax=ax)
    else:
        sns.lineplot(data=data['SOL'], ax=ax)

selected_start_date = '2020-05-30'
selected_end_date = '2021-01-01'
selected_start_date2 = '2021-03-01'
selected_end_date2 = '2021-07-25'
selected_start_date3 = '2021-07-25'
selected_end_date3 = '2021-12-25'
selected_start_date4 = '2021-12-25'
selected_end_date4 = '2022-04-01'
selected_start_date5 = '2022-07-01'
selected_end_date5 = '2023-02-01'

plt.axvspan(selected_start_date, selected_end_date, color='lightcoral', alpha=0.5)
plt.axvspan(selected_start_date2, selected_end_date2, color='lightgreen', alpha=0.5)
plt.axvspan(selected_start_date3, selected_end_date3, color='lightblue', alpha=0.5)
plt.axvspan(selected_start_date4, selected_end_date4, color='yellow', alpha=0.5)
plt.axvspan(selected_start_date5, selected_end_date5, color='lightpink', alpha=0.5)

ax.set_ylabel('Mean SOL', fontsize=10)
ax.set_xlabel('Date', fontsize=10)
ax.set_title('Monthly mean sum of lights (SOL) with timespan of COVID-19 peaks')
ax.legend()
plt.show()


# In[ ]:


# Generating a time series for the monthly deviation of provincial SOL from their three-year average

# Retrieving NTL data of provinces
provinces = ['Guimaras', 'Iloilo', 'Biliran', 'Leyte', 'Saranggani', 'South Cotabato', 'Compostela', 'Davao del Norte', 'Zamboanga Sibugay', 'Zamboanga Del Sur', 'Dinagat', 'Surigao Del Norte', 'Maguindanao', 'Shariff Kabunsuan', 'Abra', 'Apayao', 'Benguet', 'Ifugao', 'Kalinga', 'Mountain Province', 'Metropolitan Manila', 'Ilocos Norte', 'Ilocos Sur', 'La Union', 'Pangasinan', 'Batanes', 'Cagayan', 'Isabela', 'Nueva Vizcaya', 'Quirino', 'Albay', 'Camarines Norte', 'Camarines Sur', 'Catanduanes', 'Masbate', 'Sorsogon', 'Aklan', 'Antique', 'Capiz', 'Negros Occidental', 'Bohol', 'Cebu', 'Negros Oriental', 'Siquijor', 'Eastern Samar', 'Northern Samar', 'Southern Leyte', 'Samar', 'Agusan Del Norte', 'Agusan Del Sur', 'Surigao Del Sur', 'Lanao Del Sur', 'Sulu', 'Tawi-tawi', 'Basilan', 'Zamboanga Del Norte', 'Bukidnon', 'Camiguin', 'Misamis Occidental', 'Misamis Oriental', 'Lanao Del Norte', 'Davao Del Sur', 'Davao Oriental', 'North Cotabato', 'Sultan Kudarat', 'Bataan', 'Bulacan', 'Nueva Ecija', 'Pampanga', 'Tarlac', 'Zambales', 'Aurora', 'Batangas', 'Cavite', 'Laguna', 'Quezon', 'Rizal', 'Marinduque', 'Mindoro Occidental', 'Mindoro Oriental', 'Palawan', 'Romblon']
def get_province_monthly_sol(province):
    viirs = ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG").filterDate('2020-01-30', '2023-04-21')
    prov = ee.Feature(ee.FeatureCollection("FAO/GAUL/2015/level2").filter(ee.Filter.eq('ADM2_NAME', province)).first()).geometry()

    def get_prov_sol(img):
        sol = img.reduceRegion(reducer=ee.Reducer.sum(), geometry=prov, scale=500, maxPixels=1e9).get('avg_rad')
        return img.set('date', img.date().format()).set('SOL', sol)

    prov_sol = viirs.map(get_prov_sol)
    nested_list = prov_sol.reduceColumns(ee.Reducer.toList(2), ['date', 'SOL']).values().get(0)

    soldf = pd.DataFrame(nested_list.getInfo(), columns=['date', 'SOL'])
    soldf['date'] = pd.to_datetime(soldf['date'])
    soldf = soldf.set_index('date')

    # Resample data to monthly frequency and calculate the mean
    monthly_mean_SOL = soldf['SOL'].resample('M').mean()
    return monthly_mean_SOL

# Create a dictionary to store the monthly SOL data for each province
province_data = {}
for province in provinces:
    province_data[province] = get_province_monthly_sol(province)

# Create a DataFrame from the accumulated data
monthly_provsol_df = pd.DataFrame(province_data)

# Calculate the deviation by subtracting the average SOL for each province
for province in provinces:
    monthly_provsol_df[province] -= monthly_provsol_df[province].mean()

# Plot the time series for all provinces
ax = monthly_provsol_df.plot(figsize=(15, 7))
plt.title('Monthly deviation of sum of lights from average (Jan 2020 to Apr 2023)')
plt.xlabel('Date')
plt.ylabel('Deviation')

# Add a black dashed line at y=0 (2x thicker)
ax.axhline(0, color='black', linestyle='-', linewidth=2, zorder=3)

# Add a custom legend for "Metropolitan Manila" inside the upper left corner of the plot
custom_legend = ax.lines[provinces.index('Metropolitan Manila')].get_label()
ax.legend([custom_legend], title='Province', loc='lower left')

plt.show()


# In[ ]:


# Generating a time series plot where the highlighted time spans of outbreaks are superimposed with monthly deviations of NTL

# Retrieving NTL data for provinces
provinces = ['Guimaras', 'Iloilo', 'Biliran', 'Leyte', 'Saranggani', 'South Cotabato', 'Compostela', 'Davao del Norte', 'Zamboanga Sibugay', 'Zamboanga Del Sur', 'Dinagat', 'Surigao Del Norte', 'Maguindanao', 'Shariff Kabunsuan', 'Abra', 'Apayao', 'Benguet', 'Ifugao', 'Kalinga', 'Mountain Province', 'Metropolitan Manila', 'Ilocos Norte', 'Ilocos Sur', 'La Union', 'Pangasinan', 'Batanes', 'Cagayan', 'Isabela', 'Nueva Vizcaya', 'Quirino', 'Albay', 'Camarines Norte', 'Camarines Sur', 'Catanduanes', 'Masbate', 'Sorsogon', 'Aklan', 'Antique', 'Capiz', 'Negros Occidental', 'Bohol', 'Cebu', 'Negros Oriental', 'Siquijor', 'Eastern Samar', 'Northern Samar', 'Southern Leyte', 'Samar', 'Agusan Del Norte', 'Agusan Del Sur', 'Surigao Del Sur', 'Lanao Del Sur', 'Sulu', 'Tawi-tawi', 'Basilan', 'Zamboanga Del Norte', 'Bukidnon', 'Camiguin', 'Misamis Occidental', 'Misamis Oriental', 'Lanao Del Norte', 'Davao Del Sur', 'Davao Oriental', 'North Cotabato', 'Sultan Kudarat', 'Bataan', 'Bulacan', 'Nueva Ecija', 'Pampanga', 'Tarlac', 'Zambales', 'Aurora', 'Batangas', 'Cavite', 'Laguna', 'Quezon', 'Rizal', 'Marinduque', 'Mindoro Occidental', 'Mindoro Oriental', 'Palawan', 'Romblon']
def get_province_monthly_sol(province):
    viirs = ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG").filterDate('2020-01-30', '2023-04-21')
    prov = ee.Feature(ee.FeatureCollection("FAO/GAUL/2015/level2").filter(ee.Filter.eq('ADM2_NAME', province)).first()).geometry()

    def get_prov_sol(img):
        sol = img.reduceRegion(reducer=ee.Reducer.sum(), geometry=prov, scale=500, maxPixels=1e9).get('avg_rad')
        return img.set('date', img.date().format()).set('SOL', sol)

    prov_sol = viirs.map(get_prov_sol)
    nested_list = prov_sol.reduceColumns(ee.Reducer.toList(2), ['date', 'SOL']).values().get(0)

    soldf = pd.DataFrame(nested_list.getInfo(), columns=['date', 'SOL'])
    soldf['date'] = pd.to_datetime(soldf['date'])
    soldf = soldf.set_index('date')

    # Resample data to monthly frequency and calculate the mean
    monthly_mean_SOL = soldf['SOL'].resample('M').mean()
    return monthly_mean_SOL

# Create a dictionary to store the monthly SOL data for each province
province_data = {}
for province in provinces:
    province_data[province] = get_province_monthly_sol(province)

# Create a DataFrame from the accumulated data
monthly_provsol_df = pd.DataFrame(province_data)

# Calculate the deviation by subtracting the average SOL for each province
for province in provinces:
    monthly_provsol_df[province] -= monthly_provsol_df[province].mean()

ax = monthly_provsol_df.plot(figsize=(15, 7))
plt.title('Monthly deviation of sum of lights from average (Jan 2020 to Apr 2023)')
plt.xlabel('Date')
plt.ylabel('Deviation')

ax.axhline(0, color='black', linestyle='-', linewidth=2, zorder=3)

# Add a custom legend for "Metropolitan Manila" inside the upper left corner of the plot
custom_legend = ax.lines[provinces.index('Metropolitan Manila')].get_label()
ax.legend([custom_legend], title='Province', loc='lower left')

start_dates = ['2020-05-30', '2021-03-01', '2021-07-25', '2021-12-25', '2022-07-01']
end_dates = ['2021-01-01', '2021-07-25', '2021-12-25', '2022-04-01', '2023-02-01']
colors = ['lightcoral', 'lightgreen', 'lightblue', 'yellow', 'lightpink']

for i, (start_date, end_date) in enumerate(zip(start_dates, end_dates)):
    ax.axvspan(start_date, end_date, color=colors[i], alpha=0.5)

plt.show()


# ## Generating time series plots for top 10 provinces with highest cases

# In[ ]:


# Generating time series plots for top 10 provinces with highest cases
# For NCR, change 'ProvRes' to 'RegionRes'
# Metropolitan Manila, Cavite, Laguna, Rizal, Cebu, Bulacan, Batangas, Davao Del Sur, Pampanga, Iloilo

# Retrieving COVID-19 data of province
rank_prov = 'PAMPANGA' # Replaced with desired province, ensure upper case

prov_trend = frame[frame['ProvRes'] == rank_prov]
prov_trend = prov_trend.sort_values(by='DateRepConf', ascending = True, inplace = False, kind = 'quicksort', na_position = 'last')
prov_covtrend = prov_trend['DateRepConf'].value_counts(ascending = True)

prov_covtrend.sort_index(axis = 0)


# Retrieving NTL Data for province
province = "Pampanga" # Sentence case for province when using VIIRS due to GAUL Record
viirs = ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG").filterDate('2020-3-08','2023-4-21')
prov = ee.Feature(ee.FeatureCollection("FAO/GAUL/2015/level2").filter(ee.Filter.eq('ADM2_NAME', province)).first()).geometry()
provaoi = ee.Feature(ee.FeatureCollection("FAO/GAUL/2015/level2").filter(ee.Filter.eq('ADM2_NAME', province)))

def get_prov_sol(img):
    sol = img.reduceRegion(reducer=ee.Reducer.sum(), geometry=prov, scale=500, maxPixels=1e9).get('avg_rad')
    return img.set('date', img.date().format()).set('SOL',sol)

prov_sol = viirs.map(get_prov_sol)
nested_list = prov_sol.reduceColumns(ee.Reducer.toList(2), ['date','SOL']).values().get(0)

soldf = pd.DataFrame(nested_list.getInfo(), columns=['date','SOL'])
soldf['date'] = pd.to_datetime(soldf['date'])
soldf = soldf.set_index('date')

# Plotting and superimposing COVID-19 and mean SOL
prov_trend['DateRepConf'] = pd.to_datetime(prov_trend['DateRepConf'])
prov_trend = prov_trend.sort_values(by='DateRepConf', ascending=True)
prov_covtrend = prov_trend['DateRepConf'].value_counts().sort_index()

fig, ax = plt.subplots(figsize=(15, 7))

ax.plot(prov_covtrend.index, prov_covtrend.values, linestyle='-', color='red', label='Recorded number of cases')
ax.set_xlabel('Date')
ax.set_ylabel('Recorded number of COVID-19 cases', fontsize=10, color='red')
ax.set_title('COVID-19 cases and monthly sum of lights for ' + province, fontsize=15)
plt.xticks(rotation=0)

ax2 = ax.twinx()
sns.lineplot(data=soldf, ax=ax2, color='blue')
ax2.set_ylabel('Mean Sum of Lights (SOL)', fontsize=10, color='blue')
ax.legend(loc="upper left", bbox_to_anchor=(0.05, 0.92))
ax2.legend(loc="upper left", bbox_to_anchor=(0.05, 0.85))

plt.tight_layout()
plt.show()


# ## Generating line plot for yearly mean SOL

# In[ ]:


# Generating a combined line plot for the mean SOL of the Philippines and Metro Manila per year

# Retrieving NTL data for the Philippines
def get_philippines_sol(year):
    start_date = f'{year}-01-01'
    end_date = f'{year}-12-31'
    
    viirs = ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG").filterDate(start_date, end_date)
    philippines = ee.Feature(ee.FeatureCollection("FAO/GAUL/2015/level0").filter(ee.Filter.eq('ADM0_NAME', 'Philippines')).first()).geometry()

    def get_philippines_sol_per_month(img):
        sol = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=philippines, scale=500, maxPixels=1e9).get('avg_rad')
        return img.set('month', img.date().format('MM')).set('SOL', sol)

    philippines_sol = viirs.map(get_philippines_sol_per_month)
    nested_list = philippines_sol.reduceColumns(ee.Reducer.toList(2), ['month', 'SOL']).values().get(0)

    soldf = pd.DataFrame(nested_list.getInfo(), columns=['month', 'SOL'])
    soldf['month'] = soldf['month'].astype(int)
    soldf['SOL'] = soldf['SOL'] * 1e9 / 1e4
    
    return soldf

def get_province_sol(province, year):
    start_date = f'{year}-01-01'
    end_date = f'{year}-12-31'
    
    viirs = ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG").filterDate(start_date, end_date)
    prov = ee.Feature(ee.FeatureCollection("FAO/GAUL/2015/level2").filter(ee.Filter.eq('ADM2_NAME', province)).first()).geometry()

    def get_prov_sol(img):
        sol = img.reduceRegion(reducer=ee.Reducer.sum(), geometry=prov, scale=500, maxPixels=1e9).get('avg_rad')
        return img.set('date', img.date().format()).set('SOL', sol)

    prov_sol = viirs.map(get_prov_sol)
    nested_list = prov_sol.reduceColumns(ee.Reducer.toList(2), ['date', 'SOL']).values().get(0)

    soldf = pd.DataFrame(nested_list.getInfo(), columns=['date', 'SOL'])
    soldf['date'] = pd.to_datetime(soldf['date'])
    soldf = soldf.set_index('date')

    return soldf

# Specify the years
years = [2017, 2018, 2019, 2020, 2021, 2022, 2023]

# Create a list to store the results for the Philippines
results_philippines = []

# Loop through each year and append the results to the list for the Philippines
for year in years:
    philippines_data = get_philippines_sol(year)
    monthly_average = philippines_data.pivot_table(index='month', values='SOL', aggfunc='mean')
    overall_average = monthly_average['SOL'].mean()
    results_philippines.append({'Year': year, 'Overall Average SOL': overall_average})

# Convert the list to a DataFrame for the Philippines
results_philippines_df = pd.DataFrame(results_philippines)

# Create an empty list to store DataFrames for Metro Manila
dfs_metro_manila = []

# Loop through each year and append the results to the list for Metro Manila
for year in range(2017, 2024):
    metro_manila_data = get_province_sol('Metropolitan Manila', year)
    monthly_sols = metro_manila_data.resample('M').sum()
    average_sols_per_year = monthly_sols.resample('Y').mean()
    df_year = pd.DataFrame({'Year': [year], 'Average_SOL': [average_sols_per_year['SOL'].iloc[0]]})
    dfs_metro_manila.append(df_year)

# Concatenate all DataFrames into the final result for Metro Manila
result_df_metro_manila = pd.concat(dfs_metro_manila, ignore_index=True)

# Plotting the line graph for both Philippines and Metro Manila with a y-axis break
fig = plt.figure(figsize=(10, 6))
bax = brokenaxes(ylims=((0, 2500), (40000, 65000)), hspace=.05)
bax.plot(results_philippines_df['Year'], results_philippines_df['Overall Average SOL'], marker='o', linestyle='-', color='orange', label='Philippines')
bax.plot(result_df_metro_manila['Year'], result_df_metro_manila['Average_SOL'], marker='o', linestyle='-', color='b', label='Metro Manila')

bax.set_xlabel('Year')
bax.set_ylabel('Mean SOL (nW/cm²/sr)', labelpad=40)
bax.set_title('Yearly mean SOL of the Philippines and Metro Manila (2017-2023)')
bax.legend()
plt.show()


# References:  
# [1] World Bank - Light Every Night was accessed on September 2023 from https://registry.opendata.aws/wb-light-every-night.
