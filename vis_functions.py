#importing the libraries
import pandas as pd
import numpy as np
import scipy.signal
import os
import plotly.graph_objects as go
import matplotlib.dates as mdates
import mysql.connector as mysql

#Connecting to Digital Demand database
conn = mysql.connect(
    user='Elbarto',
    password='aLesa2T6nfNwSpwJAP8tKUpyLuRC8xtg',
    host='mysqldatabase.cmi5f1vp8ktf.us-east-1.rds.amazonaws.com',
    database='sandbox',
    port=3306
    )

query = """
select * from digital_demand
where gt_category=13
and country = 'DE'
and date > '2021-12-31';
"""

#NOW WE ARE CREATING df_raw simply from digital demand database
df_raw= pd.read_sql_query(query, conn, parse_dates =['date'])

#always close the connection
conn.close()

#add indexing function
def add_indexing(df,var,index_date):
    '''
    Adding indexes to the var in a dataframe 
    so that we don't get values between 0 to 1
    and instead obtain results in our own scale
    
    i.e if vl_value in 1st Jan 2022 (index_date) in 0.01
    then we can set vl_value_ref to the vl_value in that row
    and then if we find a vl_value in any other date 
    we compare this to vl_value_ref by scaling it using vl_value_ref 
    (vl_value/vl_value_ref * 100) to obtain value for vl_value_index column
    (always grouped by keyword, country, category)
    
    Parameters:
        df(dataframe)
        
        var(str) : string of a numeric column of the dataframe 
        
        index_date(str): date string
    
    Returns:
        df: dataframe with a new column which is called var_index
        i.e vl_value_index
    '''
    var_ref = var +'_ref'                                      #variable for index computation
    var_new = var +'_index'                                    #new index variable to be added to df  
    df_ref = df[df['date']==index_date]                        #create reference df with values from indexdate
    df_ref = df_ref.rename(columns={var : var_ref})            #rename to avoid confusion
    #Add values of indexdate to original dataframe and compute index values
    df_w_index = pd.merge(df, df_ref[['keyword',
                                      'country',
                                      'gt_category',
                                      var_ref]],
                          how="left",
                          on=['keyword',
                              'country',
                              'gt_category'
                             ])
    df_w_index[var_new] = (df_w_index[var]/df_w_index[var_ref])*100
    return df_w_index


#indexing avg function
def add_indexing_by_avg(df,var):
    '''
    Adding indexes to the var in a dataframe 
    so that we don't get values between 0 to 1
    and instead obtain results in our own scale
    (always grouped by keyword, country, category)
    
    i.e 
    here we are obtaining the mean value for a given keyword, country, category combination
    and using that as a reference to create values for the var_index_avg column
    
    Parameters:
        df(dataframe)
        
        var(str) : string of a numeric column of the dataframe 
    
    Returns:
        df: dataframe with a new column which is called var_index_avg
        i.e vl_value_index_avg
    '''
    var_ref = var +'_ref_avg'
    var_new = var +'_index_avg'
    df_index = df.copy()
    df_index[var_ref] = df_index.groupby(['keyword',
                                          'country',
                                          'gt_category'
                                         ])[var].transform(lambda x: x.mean())    #compute moving average
    df_index[var_new] = (df_index[var]/df_index[var_ref])*100
    return df_index



#moving average function
def add_ma(df,var,window):
    '''
    Adding moving avg column to the dataframe
    (always grouped by keyword, country, category)
        
    Parameters:
        df(dataframe)
        
        var(str) : string of a numeric column of the dataframe
        
        window(int): moving average window 
        i.e if 7 (will calculate from the 7th day and obtain NAN for days 1 to 6)
        
    
    Returns:
        df: dataframe with a new column which is called var_ma_{windowint}
        i.e vl_value_ma7
    '''

    
    var_new = var + '_ma'                                       #new ma variable to be added to df
    df = df.sort_values(by=['keyword',
                            'gt_category',
                            'country',
                            'date'
                           ])
    df[var_new] = df.groupby(['keyword',
                              'country',
                              'gt_category'
                             ])[var].transform(lambda x: x.rolling(window).mean())    #compute moving average
    
    df = df.rename(columns={var_new: var_new+str(window)})
    return df


#standard deviation function
def add_std(df,var,window):
    '''
    Adding standard_deviation of the moving average to the dataframe in a new column
    (always grouped by keyword, country, category)
    
    Parameters:
        df(dataframe)
        
        var(str) : string of a numeric column of the dataframe
        
        window(int): moving average window 
        i.e if 7 (will calculate from the 7th day and obtain NAN for days 1 to 6)
        
    
    Returns:
        df: dataframe with a new column which is called var_std_{windowint}
        i.e vl_value_std7
    '''
        
    var_new = var + '_std'                                       #new ma variable to be added to df
    df = df.sort_values(by=['keyword',
                            'gt_category',
                            'country',
                            'date'
                           ])
    df[var_new] = df.groupby(['keyword',
                              'country',
                              'gt_category'
                             ])[var].transform(lambda x: 2*x.rolling(window).std())    #compute moving average
    df = df.rename(columns={var_new: var_new+str(window)})
    return df


#smoother function
def add_smoother(df,var,cutoff):
    '''
    Adding smooth values for var in the dataframe in a new column
    (always grouped by keyword, country, category)
    
    Parameters:
        df(dataframe)
        
        var(str) : string of a numeric column of the dataframe
        
        cutoff(float): cutoff value for smoothing and expects values in between 0 to 1
        degree of smoothing 
        i.e we are currently choosing 0.02
        refernce: https://swharden.com/blog/2020-09-23-signal-filtering-in-python/
        
    
    Returns:
        df: dataframe with a new column which is called var_smooth
        i.e vl_value_smooth
    '''
    b, a = scipy.signal.butter(3, cutoff)
    var_new = var + '_smooth'                                       #new ma variable to be added to df
    df = df.sort_values(by=['keyword',
                            'gt_category',
                            'country',
                            'date'
                           ])
    df[var_new] = df.groupby(['keyword',
                              'country',
                              'gt_category'
                             ])[var].transform(lambda x: scipy.signal.filtfilt(b, a, x))    #compute moving average
    return df




#function that produces and saves the vis
# def single(key,geo,cat,startdate,index,indexdate,font_use,out_type):
#     '''
#     Creating a single time series visualization that includes raw_timeseries, trend, moving avg, smoothed trendlines
    
#     Parameters:
#         key(str): keyword in digital demand dataframe
        
#         geo(str): country value in digital demand dataframe
        
#         cat(int) : category value in digital demand dataframe
        
#         startdate(str): gives us the start value for the visualization
#         i.e '2010-01-01' - the vis would start at 1st Jan 2010
        
#         index(bool): whether you want to add an indexed column to the dataframe and plot the column as well
        
#         indexdate(str): reference for index column
        
#         font_use(str): font you want in the plot
        
#         out_type(str): the format of the output that you want
#         i.e 'svg', 'html', 'png'
    
#     Returns:
#         a local copy of the visualization in the format you want (svg, png etc)
#         saves it in desktop
#     '''
    
#     df_key = df_raw[(df_raw.keyword == f'{params.get("key")}')\
#                     &(df_raw.country == f'{params.get("geo")}')\
#                     &(df_raw.gt_category == int(f'{params.get("cat")}'))]
#     if params.get("index")==True: 
#         df_key = add_indexing(df_key,'vl_value',f'{params.get("indexdate")}')
#         var_new = 'vl_value_index'
#     else:
#         var_new = 'vl_value'
#         #running the functions we created to create moving average, smoother
#     df_key = add_ma(df_key,var_new,14)
#     df_key = add_smoother(df_key,var_new,0.02) 
#     df = df_key[df_key.date>=f'{params["startdate"]}']
#     fig = go.Figure()
#     fig.add_trace(
#         go.Scatter( 
#             x=df.date, 
#             y=df[var_new],
#             name='original', 
#             mode='lines',
#             opacity = 0.3,
#             line=dict(color='#024D83',
#                       width=4),
#             showlegend=True
#     ))
#     #creating the trendline values
#     df_trend = df[['date',var_new]]         #i.e we need date and vl_value 
#     df_trend0 = df_trend.dropna()           #dropping 0 because trendlines can't cope without numeric values
#     x_sub = df_trend0.date    
#     y_sub = df_trend0[var_new]
#     x_sub_num = mdates.date2num(x_sub)      #transforming dates to numeric values, necessary for polynomial fitting
#     z_sub = np.polyfit(x_sub_num, y_sub, 1) #polynomial fitting
#     p_sub = np.poly1d(z_sub)
#     #adding the trendline trace
#     fig.add_trace(
#         go.Scatter( 
#             x=x_sub, 
#             y=p_sub(x_sub_num), 
#             name='trend', 
#             mode='lines',
#             opacity = 1,
#             line=dict(color='green',
#                       width=4,
#                       dash='dash')
#     ))
#     #adding the 2 week's moving avg trace
#     fig.add_trace(
#         go.Scatter( 
#             x=df.date, 
#             y=df[var_new+'_ma'+str(14)],
#             name=var_new+'_ma'+str(14), 
#             mode='lines',
#             opacity = 1,
#             line=dict(color='red',
#                       width=4),
#             showlegend=True
#     ))
#     #adding the smoothed trace
#     fig.add_trace(
#         go.Scatter( 
#             x=df.date, 
#             y=df[var_new+'_smooth'],
#             name='smoothed', 
#             mode='lines',
#             opacity = 1,
#             line=dict(color='purple',
#                       width=6),
#             showlegend=True
#     ))
#     fig.update_layout(
#         xaxis={'title': None,
#                'titlefont':{'color':'#BFBFBF', 
#                             'family': font_use},
#                'tickfont':{'color':'#002A34',
#                            'size':30, 
#                            'family': font_use},
#                'gridcolor': '#4A4A4A',
#                'linecolor': '#000000',
#                'showgrid':False},
#         yaxis={'title': 'Digital Demand'  ,
#                'titlefont':{'color':'#002A34',
#                             'size':50, 
#                             'family': font_use},
#                'tickfont':{'color':'#002A34',
#                            'size':30, 
#                            'family': font_use},
#                'showgrid':False,
#                'zeroline':False},
#         margin={'l': 170, 
#                 'b': 150, 
#                 't': 150,
#                 'r': 40},
#         title={'text': None, 
#                'font':{'color':'#000000', 
#                        'size':40},
#                'yanchor':"top",
#                'xanchor':"center"},
#         legend={'font':{'size':20, 
#                         'color':'#333',
#                         'family': font_use},
#                 'yanchor':"top",
#                 'xanchor':"center",
#                 'y':0.9,
#                 'x':.95,
#                 'orientation':'v',
#                 },
#         template = 'none',
#         hovermode='x unified',
#         width = 1920,
#         height = 1080     
#     )
#     if out_type == 'svg':
#         fig.write_image(os.path.expanduser(f"~/Desktop/{key}_single_timeseries.svg"))
#     elif out_type == 'html':
#         fig.write_html(os.path.expanduser(f"~/Desktop/{key}_single_timeseries.html"))
#     else:
#         fig.write_image(os.path.expanduser(f"~/Desktop/plot.png"))
        
#     return 'vis completed'


def test():
    #print('hello')
    return 'hello'

#test()
