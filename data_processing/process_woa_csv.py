import gzip 
import numpy as np 
import pandas as pd 
import sys 

# FILE_PATH = sys.argv[1]



def read_process(FILE_PATH, month=1, chemical='dissolved_oxygen'):

    df = pd.read_csv(FILE_PATH,  compression='gzip',skiprows=1, header=0)

    #Changing columns names
    df.columns.values[0] = 'lat' 
    df.columns.values[1] = 'lon'
    df.columns.values[2] = '0'

    desired_depths = ['0', '5', '10', '15', '20', '25', '30', '35', '40', '45',
       '50', '55', '60', '65', '70', '75', '80', '85', '90', '95', '100',
       '125', '150', '175', '200', '225', '250', '275', '300']
    
    # This data has a grid with in every 5°, and middle point of the grids. 
    # This process push it to the left of the grid

    df['lat'] = df['lat'].apply(lambda x: x-2.5)
    df['lon'] = df['lon'].apply(lambda x: x-2.5)

    # Adding the month number to the dataframe
    mon = month*np.ones(len(df))
    df['month'] = mon.astype(int)

    # Taking the mean of dissolved oxgyen until the depth of 300m

    df[chemical] = df[desired_depths].mean(axis=1)

    df_temp = df[['lat','lon','month',chemical]]
    df_last = df_temp[(df_temp['lat'] <35)&(df_temp['lat'] >= -15 )& (df_temp['lon'] < 0)&(df_temp['lon'] >= -100 )]

    return df_last 


if __name__=="__main__":
    df = pd.DataFrame()
    
    #Oxygen

    # for i in range(1,10):
    #     FILE_PATH = f'woa23_o_all_5.00_csv/months/woa23_all_o0{i}mn5d.csv.gz'

    #     df_temp = read_process(FILE_PATH, i)

    #     df = pd.concat([df,df_temp])

    # for i in range(10,13):
    #     FILE_PATH = f'woa23_o_all_5.00_csv/months/woa23_all_o{i}mn5d.csv.gz'

    #     df_temp = read_process(FILE_PATH, i)

    #     df = pd.concat([df,df_temp])

    # df.to_csv('oxygen_mn_montly.csv', index=False)
    # print(df)

    #Silicate

    # for i in range(1,10):
    #     FILE_PATH = f'woa23_i_all_5.00_csv/woa23_all_i0{i}mn5d.csv.gz'

    #     df_temp = read_process(FILE_PATH=FILE_PATH, month=i, chemical='silicate')

    #     df = pd.concat([df,df_temp])

    # for i in range(10,13):
    #     FILE_PATH = f'woa23_i_all_5.00_csv/woa23_all_i{i}mn5d.csv.gz'

    #     df_temp = read_process(FILE_PATH=FILE_PATH, month=i, chemical='silicate')

    #     df = pd.concat([df,df_temp])

    # df.to_csv('Silicate_mn_montly.csv', index=False)
    # print(df)

    #Nitrate

    # for i in range(1,10):
    #     FILE_PATH = f'woa23_n_all_5.00_csv/woa23_all_n0{i}mn5d.csv.gz'

    #     df_temp = read_process(FILE_PATH=FILE_PATH, month=i, chemical='nitrate')

    #     df = pd.concat([df,df_temp])

    # for i in range(10,13):
    #     FILE_PATH = f'woa23_n_all_5.00_csv/woa23_all_n{i}mn5d.csv.gz'

    #     df_temp = read_process(FILE_PATH=FILE_PATH, month=i, chemical='nitrate')

    #     df = pd.concat([df,df_temp])

    # df.to_csv('nitrate_mn_montly.csv', index=False)
    # print(df)
    #Phosphate 

    for i in range(1,10):
        FILE_PATH = f'woa23_p_all_5.00_csv/woa23_all_p0{i}mn5d.csv.gz'

        df_temp = read_process(FILE_PATH=FILE_PATH, month=i, chemical='phosphate')

        df = pd.concat([df,df_temp])

    for i in range(10,13):
        FILE_PATH = f'woa23_p_all_5.00_csv/woa23_all_p{i}mn5d.csv.gz'

        df_temp = read_process(FILE_PATH=FILE_PATH, month=i, chemical='phosphate')

        df = pd.concat([df,df_temp])

    df.to_csv('phosphate_mn_montly.csv', index=False)
    print(df)