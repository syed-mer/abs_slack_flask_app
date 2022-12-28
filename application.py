#LOADING THE LIBRARIES
import os,shutil,glob
import slack
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from slackeventsapi import SlackEventAdapter
from pathlib import Path              #to load environment variable
from dotenv import load_dotenv        #to load environment variable
import mysql.connector as mysql
import numpy as np
import pandas as pd
import requests                       #to obtain the payload text
import sys
import getopt
#datetime imports
from datetime import date
from datetime import datetime as dt
from datetime import timedelta as delta
from datetime import timedelta
import numexpr
#FLASK
from flask import Flask, request, Response
from threading import Thread #used to make background process work for slackbot
import json
import base64
#Visualization libraries
import seaborn as sns
import plotly.graph_objects as go
import plotly
import plotly.io as pio
import plotly.offline as pyo
from plotly.subplots import make_subplots
import plotly.express as px
#import plotly.graph_objects as go
#from plotly.subplots import make_subplots
from PIL import Image
import matplotlib.dates as mdates
import matplotlib
#from pylab import *
#SMOOTHING THE GRAPH
import scipy.signal
#IPYTHON WIDGET IMPORTS
#import ipywidgets as widgets
#from IPython.display import display
#from ipywidgets import interact, Layout 
#from ipywidgets import IntSlider 
import warnings
#IMPORTING FUNCTIONS FROM vis_functions.py
from vis_functions import *
from slack.signature import SignatureVerifier



#os.environ
#setting the path to environment variable (.env file) and loading it
#this is done to protect the slack_bot_token 
env_path = Path('.') / '.env'
load_dotenv(dotenv_path = env_path)

app = Flask(__name__)
#1st param is signing secret stored inside .env file
#2nd param is root where we want to send different events to
#3rd param is where we are sending the events
slack_event_adapter = SlackEventAdapter(
    os.environ['SIGNING_SECRET'], '/slack/events', app)

#getting slack_bot_token from our stored environment variable
client = WebClient(token=os.environ['SLACK_TOKEN'])
#obtains bot id
BOT_ID = client.api_call("auth.test")['user_id'] #gives us the bot id

signature_verifier = SignatureVerifier(os.environ["SIGNING_SECRET"])

@slack_event_adapter.on('message')
def message(payload):
    #count=0
    #if count == 0:
    event = payload.get('event', {})
    channel_id = event.get('channel')
    user_id = event.get('user')
    text = event.get('text')
    
    message_id = event.get('ts')
    
        
    data = request.form
    #print(data)
    user_id = data.get('user_id')
    channel_id = data.get('channel_id')
    
    client.chat_postMessage(channel='#asb_dd_top10_changes', 
                                        text=f"Visualization922: {event}"
                                        )
    
    
def backgroundworker(text,response_url):

    # your task
    
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
    and country = 'DE';
    """
    
    #NOW WE ARE CREATING df_raw simply from digital demand database
    df_raw= pd.read_sql_query(query, conn, parse_dates =['date'])

    #always close the connection
    conn.close()
    
    #This block is triggered when keyword is not in digital demand 
    missing_kw_block = [
		{
			"type": "divider"
		},
		{
			"type": "divider"
		},
		{
			"type": "section",
			"text": {
				"type": "mrkdwn",
				"text": "*MISSING KEYWORD*\n\n Keyword not in Digital Demand Database. \n\n"
			},
			"accessory": {
				"type": "image",
				"image_url": "https://www.publicdomainpictures.net/pictures/280000/velka/not-found-image-15383864787lu.jpg",
				"alt_text": "alt text for image"
			}
		},
		{
			"type": "divider"
		},
		{
			"type": "section",
			"text": {
				"type": "mrkdwn",
				"text": "_Please try the command again with a differerent keyword._ "
			}
		}
	]
    
    #NEW ADDITION posts message when keyword isn't available
    if text.lower() not in df_raw.keyword.unique().tolist():
        client.chat_postMessage(channel='#asb_dd_top10_changes', 
                                            text=" ",
                                            blocks=missing_kw_block
                                            )
    else:
        pass
    
    #we are creating manuals parameter dictionary for function values at the moment
    params = {'key': f'{text.lower()}',
              'geo': 'DE',
              'cat': 13,
              'startdate': '2022-01-01',
              'index': False,
              'indexdate': '2022-08-01',
              'font_use': 'Roboto Mono Light for Powerline',
              'out_type': 'png'
             }
    
    #function that produces and saves the vis
    def single(key,geo,cat,startdate,index,indexdate,font_use,out_type):
        '''
        Creating a single time series visualization that includes raw_timeseries, trend, moving avg, smoothed trendlines
        
        Parameters:
            key(str): keyword in digital demand dataframe
            
            geo(str): country value in digital demand dataframe
            
            cat(int) : category value in digital demand dataframe
            
            startdate(str): gives us the start value for the visualization
            i.e '2010-01-01' - the vis would start at 1st Jan 2010
            
            index(bool): whether you want to add an indexed column to the dataframe and plot the column as well
            
            indexdate(str): reference for index column
            
            font_use(str): font you want in the plot
            
            out_type(str): the format of the output that you want
            i.e 'svg', 'html', 'png'
        
        Returns:
            a local copy of the visualization in the format you want (svg, png etc)
            saves it in desktop
        '''
        
        df_key = df_raw[(df_raw.keyword == f'{params.get("key")}')\
                        &(df_raw.country == f'{params.get("geo")}')\
                        &(df_raw.gt_category == int(f'{params.get("cat")}'))]
        if params.get("index")==True: 
            df_key = add_indexing(df_key,'vl_value',f'{params.get("indexdate")}')
            var_new = 'vl_value_index'
        else:
            var_new = 'vl_value'
            #running the functions we created to create moving average, smoother
        df_key = add_ma(df_key,var_new,14)
        df_key = add_smoother(df_key,var_new,0.02) 
        df = df_key[df_key.date>=f'{params["startdate"]}']
        fig = go.Figure()
        fig.add_trace(
            go.Scatter( 
                x=df.date, 
                y=df[var_new],
                name='original', 
                mode='lines',
                opacity = 0.3,
                line=dict(color='#024D83',
                          width=4),
                showlegend=True
        ))
        #creating the trendline values
        df_trend = df[['date',var_new]]         #i.e we need date and vl_value 
        df_trend0 = df_trend.dropna()           #dropping 0 because trendlines can't cope without numeric values
        x_sub = df_trend0.date    
        y_sub = df_trend0[var_new]
        x_sub_num = mdates.date2num(x_sub)      #transforming dates to numeric values, necessary for polynomial fitting
        z_sub = np.polyfit(x_sub_num, y_sub, 1) #polynomial fitting
        p_sub = np.poly1d(z_sub)
        #adding the trendline trace
        fig.add_trace(
            go.Scatter( 
                x=x_sub, 
                y=p_sub(x_sub_num), 
                name='trend', 
                mode='lines',
                opacity = 1,
                line=dict(color='green',
                          width=4,
                          dash='dash')
        ))
        #adding the 2 week's moving avg trace
        fig.add_trace(
            go.Scatter( 
                x=df.date, 
                y=df[var_new+'_ma'+str(14)],
                name=var_new+'_ma'+str(14), 
                mode='lines',
                opacity = 1,
                line=dict(color='red',
                          width=4),
                showlegend=True
        ))
        #adding the smoothed trace
        fig.add_trace(
            go.Scatter( 
                x=df.date, 
                y=df[var_new+'_smooth'],
                name='smoothed', 
                mode='lines',
                opacity = 1,
                line=dict(color='purple',
                          width=6),
                showlegend=True
        ))
        fig.update_layout(
            xaxis={'title': None,
                   'titlefont':{'color':'#BFBFBF', 
                                'family': font_use},
                   'tickfont':{'color':'#002A34',
                               'size':30, 
                               'family': font_use},
                   'gridcolor': '#4A4A4A',
                   'linecolor': '#000000',
                   'showgrid':False},
            yaxis={'title': 'Digital Demand'  ,
                   'titlefont':{'color':'#002A34',
                                'size':50, 
                                'family': font_use},
                   'tickfont':{'color':'#002A34',
                               'size':30, 
                               'family': font_use},
                   'showgrid':False,
                   'zeroline':False},
            margin={'l': 170, 
                    'b': 150, 
                    't': 150,
                    'r': 40},
            title={'text': f'{text}'.capitalize(), 
                   'font':{'color':'#000000', 
                           'size':40,
                           'family': font_use},
                   'yanchor':"top",
                   'xanchor':"center"},
            legend={'font':{'size':20, 
                            'color':'#333',
                            'family': font_use},
                    'yanchor':"top",
                    'xanchor':"center",
                    'y':0.9,
                    'x':.95,
                    'orientation':'v',
                    },
            template = 'none',
            hovermode='x unified',
            width = 1920,
            height = 1080     
        )
        if out_type == 'svg':
            fig.write_image(os.path.expanduser(f"~/Desktop/{key}_single_timeseries.svg"))
        elif out_type == 'html':
            fig.write_html(os.path.expanduser(f"~/Desktop/{key}_single_timeseries.html"))
        else:
            fig.write_image(os.path.expanduser(f"{text}.png"))
            
        return 'vis completed'
    
    #this is running from vis_functions.py
    single(
        key = f'{text.lower()}', 
        geo = 'DE',
        cat = 13,
        startdate = '2020-01-01',
        index = False,
        indexdate = '2022-08-01',
        font_use = 'Roboto Mono Light for Powerline',
        out_type = 'png'
    )
    
    #payload is required to to send second message after task is completed
    payload = {"text":"your task is complete",
                "username": "bot"}
    
    #uploading the file to slack using bolt syntax for py
    try:
        filename=f"{text}.png"
        response = client.files_upload(channels='#asb_dd_top10_changes',
                                        file=filename,
                                        initial_comment="Visualization: ")
        assert response["file"]  # the uploaded file
    except SlackApiError as e:
        # You will get a SlackApiError if "ok" is False
        assert e.response["ok"] is False
        assert e.response["error"]  # str like 'invalid_auth', 'channel_not_found'
        print(f"Got an error: {e.response['error']}")

    requests.post(response_url,data=json.dumps(payload))
    
    


#test background worker 2
def backgroundworker2(text, init_date, response_url):

    # your task
    
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
    and country = 'DE';
    """
    
    #NOW WE ARE CREATING df_raw simply from digital demand database
    df_raw= pd.read_sql_query(query, conn, parse_dates =['date'])

    #always close the connection
    conn.close()
    
    #NEW ADDITION
    if text.lower() not in df_raw.keyword.unique().tolist():
        client.chat_postMessage(channel='#asb_dd_top10_changes', 
                                            text="Keyword not in Digital Demand Database. Please try the command again with a differenrent keyword. "
                                            )
    else:
        pass
    
    #we are creating manuals parameter dictionary for function values at the moment
    params = {'key': f'{text.lower()}',
              'geo': 'DE',
              'cat': 13,
              'startdate': f'{init_date}',
              'index': False,
              'indexdate': '2022-08-01',
              'font_use': 'Roboto Mono Light for Powerline',
              'out_type': 'png'
             }
    
    #function that produces and saves the vis
    def single(key,geo,cat,startdate,index,indexdate,font_use,out_type):
        '''
        Creating a single time series visualization that includes raw_timeseries, trend, moving avg, smoothed trendlines
        
        Parameters:
            key(str): keyword in digital demand dataframe
            
            geo(str): country value in digital demand dataframe
            
            cat(int) : category value in digital demand dataframe
            
            startdate(str): gives us the start value for the visualization
            i.e '2010-01-01' - the vis would start at 1st Jan 2010
            
            index(bool): whether you want to add an indexed column to the dataframe and plot the column as well
            
            indexdate(str): reference for index column
            
            font_use(str): font you want in the plot
            
            out_type(str): the format of the output that you want
            i.e 'svg', 'html', 'png'
        
        Returns:
            a local copy of the visualization in the format you want (svg, png etc)
            saves it in desktop
        '''
        
        df_key = df_raw[(df_raw.keyword == f'{params.get("key")}')\
                        &(df_raw.country == f'{params.get("geo")}')\
                        &(df_raw.gt_category == int(f'{params.get("cat")}'))]
        if params.get("index")==True: 
            df_key = add_indexing(df_key,'vl_value',f'{params.get("indexdate")}')
            var_new = 'vl_value_index'
        else:
            var_new = 'vl_value'
            #running the functions we created to create moving average, smoother
        df_key = add_ma(df_key,var_new,14)
        df_key = add_smoother(df_key,var_new,0.02) 
        df = df_key[df_key.date>=f'{params["startdate"]}']
        fig = go.Figure()
        fig.add_trace(
            go.Scatter( 
                x=df.date, 
                y=df[var_new],
                name='original', 
                mode='lines',
                opacity = 0.3,
                line=dict(color='#024D83',
                          width=4),
                showlegend=True
        ))
        #creating the trendline values
        df_trend = df[['date',var_new]]         #i.e we need date and vl_value 
        df_trend0 = df_trend.dropna()           #dropping 0 because trendlines can't cope without numeric values
        x_sub = df_trend0.date    
        y_sub = df_trend0[var_new]
        x_sub_num = mdates.date2num(x_sub)      #transforming dates to numeric values, necessary for polynomial fitting
        z_sub = np.polyfit(x_sub_num, y_sub, 1) #polynomial fitting
        p_sub = np.poly1d(z_sub)
        #adding the trendline trace
        fig.add_trace(
            go.Scatter( 
                x=x_sub, 
                y=p_sub(x_sub_num), 
                name='trend', 
                mode='lines',
                opacity = 1,
                line=dict(color='green',
                          width=4,
                          dash='dash')
        ))
        #adding the 2 week's moving avg trace
        fig.add_trace(
            go.Scatter( 
                x=df.date, 
                y=df[var_new+'_ma'+str(14)],
                name=var_new+'_ma'+str(14), 
                mode='lines',
                opacity = 1,
                line=dict(color='red',
                          width=4),
                showlegend=True
        ))
        #adding the smoothed trace
        fig.add_trace(
            go.Scatter( 
                x=df.date, 
                y=df[var_new+'_smooth'],
                name='smoothed', 
                mode='lines',
                opacity = 1,
                line=dict(color='purple',
                          width=6),
                showlegend=True
        ))
        fig.update_layout(
            xaxis={'title': None,
                   'titlefont':{'color':'#BFBFBF', 
                                'family': font_use},
                   'tickfont':{'color':'#002A34',
                               'size':30, 
                               'family': font_use},
                   'gridcolor': '#4A4A4A',
                   'linecolor': '#000000',
                   'showgrid':False},
            yaxis={'title': 'Digital Demand'  ,
                   'titlefont':{'color':'#002A34',
                                'size':50, 
                                'family': font_use},
                   'tickfont':{'color':'#002A34',
                               'size':30, 
                               'family': font_use},
                   'showgrid':False,
                   'zeroline':False},
            margin={'l': 170, 
                    'b': 150, 
                    't': 150,
                    'r': 40},
            title={'text': f'{text}'.capitalize(), 
                   'font':{'color':'#000000', 
                           'size':40,
                           'family': font_use},
                   'yanchor':"top",
                   'xanchor':"center"},
            legend={'font':{'size':20, 
                            'color':'#333',
                            'family': font_use},
                    'yanchor':"top",
                    'xanchor':"center",
                    'y':0.9,
                    'x':.95,
                    'orientation':'v',
                    },
            template = 'none',
            hovermode='x unified',
            width = 1920,
            height = 1080     
        )
        if out_type == 'svg':
            fig.write_image(os.path.expanduser(f"~/Desktop/{key}_single_timeseries.svg"))
        elif out_type == 'html':
            fig.write_html(os.path.expanduser(f"~/Desktop/{key}_single_timeseries.html"))
        else:
            fig.write_image(os.path.expanduser(f"{text}.png"))
            
        return 'vis completed'
    
    #this is running from vis_functions.py
    single(
        key = f'{text.lower()}', 
        geo = 'DE',
        cat = 13,
        startdate = f'{init_date}',
        index = False,
        indexdate = '2022-08-01',
        font_use = 'Roboto Mono Light for Powerline',
        out_type = 'png'
    )
    
    #payload is required to to send second message after task is completed
    payload = {"text":"your task is complete",
                "username": "bot"}
    
    #uploading the file to slack using bolt syntax for py
    try:
        filename=f"{text}.png"
        response = client.files_upload(channels='#asb_dd_top10_changes',
                                        file=filename,
                                        initial_comment="Visualization: ")
        assert response["file"]  # the uploaded file
    except SlackApiError as e:
        # You will get a SlackApiError if "ok" is False
        assert e.response["ok"] is False
        assert e.response["error"]  # str like 'invalid_auth', 'channel_not_found'
        print(f"Got an error: {e.response['error']}")

    requests.post(response_url,data=json.dumps(payload))

#test background worker 2
def backgroundworker3(text, init_date, index_date, response_url):

    # your task
    
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
    and country = 'DE';
    """

    
    #NOW WE ARE CREATING df_raw simply from digital demand database
    df_raw= pd.read_sql_query(query, conn, parse_dates =['date'])

    #always close the connection
    conn.close()
    
    #NEW ADDITION
    if text.lower() not in df_raw.keyword.unique().tolist():
        client.chat_postMessage(channel='#asb_dd_top10_changes', 
                                            text="Keyword not in Digital Demand Database. Please try the command again with a differenrent keyword. "
                                            )
    else:
        pass
    
    #we are creating manuals parameter dictionary for function values at the moment
    params = {'key': f'{text.lower()}',
              'geo': 'DE',
              'cat': 13,
              'startdate': f'{init_date}',
              'index': True,
              'indexdate': f'{index_date}',
              'font_use': 'Roboto Mono Light for Powerline',
              'out_type': 'png'
             }
    
    #function that produces and saves the vis
    def single(key,geo,cat,startdate,index,indexdate,font_use,out_type):
        '''
        Creating a single time series visualization that includes raw_timeseries, trend, moving avg, smoothed trendlines
        
        Parameters:
            key(str): keyword in digital demand dataframe
            
            geo(str): country value in digital demand dataframe
            
            cat(int) : category value in digital demand dataframe
            
            startdate(str): gives us the start value for the visualization
            i.e '2010-01-01' - the vis would start at 1st Jan 2010
            
            index(bool): whether you want to add an indexed column to the dataframe and plot the column as well
            
            indexdate(str): reference for index column
            
            font_use(str): font you want in the plot
            
            out_type(str): the format of the output that you want
            i.e 'svg', 'html', 'png'
        
        Returns:
            a local copy of the visualization in the format you want (svg, png etc)
            saves it in desktop
        '''
        
        df_key = df_raw[(df_raw.keyword == f'{params.get("key")}')\
                        &(df_raw.country == f'{params.get("geo")}')\
                        &(df_raw.gt_category == int(f'{params.get("cat")}'))]
        if params.get("index")==True: 
            df_key = add_indexing(df_key,'vl_value',f'{params.get("indexdate")}')
            var_new = 'vl_value_index'
        else:
            var_new = 'vl_value'
            #running the functions we created to create moving average, smoother
        df_key = add_ma(df_key,var_new,14)
        df_key = add_smoother(df_key,var_new,0.02) 
        df = df_key[df_key.date>=f'{params["startdate"]}']
        fig = go.Figure()
        fig.add_trace(
            go.Scatter( 
                x=df.date, 
                y=df[var_new],
                name='original', 
                mode='lines',
                opacity = 0.3,
                line=dict(color='#024D83',
                          width=4),
                showlegend=True
        ))
        #creating the trendline values
        df_trend = df[['date',var_new]]         #i.e we need date and vl_value 
        df_trend0 = df_trend.dropna()           #dropping 0 because trendlines can't cope without numeric values
        x_sub = df_trend0.date    
        y_sub = df_trend0[var_new]
        x_sub_num = mdates.date2num(x_sub)      #transforming dates to numeric values, necessary for polynomial fitting
        z_sub = np.polyfit(x_sub_num, y_sub, 1) #polynomial fitting
        p_sub = np.poly1d(z_sub)
        #adding the trendline trace
        fig.add_trace(
            go.Scatter( 
                x=x_sub, 
                y=p_sub(x_sub_num), 
                name='trend', 
                mode='lines',
                opacity = 1,
                line=dict(color='green',
                          width=4,
                          dash='dash')
        ))
        #adding the 2 week's moving avg trace
        fig.add_trace(
            go.Scatter( 
                x=df.date, 
                y=df[var_new+'_ma'+str(14)],
                name=var_new+'_ma'+str(14), 
                mode='lines',
                opacity = 1,
                line=dict(color='red',
                          width=4),
                showlegend=True
        ))
        #adding the smoothed trace
        fig.add_trace(
            go.Scatter( 
                x=df.date, 
                y=df[var_new+'_smooth'],
                name='smoothed', 
                mode='lines',
                opacity = 1,
                line=dict(color='purple',
                          width=6),
                showlegend=True
        ))
        fig.update_layout(
            xaxis={'title': None,
                   'titlefont':{'color':'#BFBFBF', 
                                'family': font_use},
                   'tickfont':{'color':'#002A34',
                               'size':30, 
                               'family': font_use},
                   'gridcolor': '#4A4A4A',
                   'linecolor': '#000000',
                   'showgrid':False},
            yaxis={'title': 'Digital Demand'  ,
                   'titlefont':{'color':'#002A34',
                                'size':50, 
                                'family': font_use},
                   'tickfont':{'color':'#002A34',
                               'size':30, 
                               'family': font_use},
                   'showgrid':False,
                   'zeroline':False},
            margin={'l': 170, 
                    'b': 150, 
                    't': 150,
                    'r': 40},
            title={'text': f'{text}'.capitalize(), 
                   'font':{'color':'#000000', 
                           'size':40,
                           'family': font_use},
                   'yanchor':"top",
                   'xanchor':"center"},
            legend={'font':{'size':20, 
                            'color':'#333',
                            'family': font_use},
                    'yanchor':"top",
                    'xanchor':"center",
                    'y':0.9,
                    'x':.95,
                    'orientation':'v',
                    },
            template = 'none',
            hovermode='x unified',
            width = 1920,
            height = 1080     
        )
        if out_type == 'svg':
            fig.write_image(os.path.expanduser(f"~/Desktop/{key}_single_timeseries.svg"))
        elif out_type == 'html':
            fig.write_html(os.path.expanduser(f"~/Desktop/{key}_single_timeseries.html"))
        else:
            fig.write_image(os.path.expanduser(f"{text}.png"))
            
        return 'vis completed'
    
    #this is running from vis_functions.py
    single(
        key = f'{text.lower()}', 
        geo = 'DE',
        cat = 13,
        startdate = f'{init_date}',
        index = True,
        indexdate = f'{index_date}',
        font_use = 'Roboto Mono Light for Powerline',
        out_type = 'png'
    )
    
    #payload is required to to send second message after task is completed
    payload = {"text":"your task is complete",
                "username": "bot"}
    
    #uploading the file to slack using bolt syntax for py
    try:
        filename=f"{text}.png"
        response = client.files_upload(channels='#asb_dd_top10_changes',
                                        file=filename,
                                        initial_comment="Visualization: ")
        assert response["file"]  # the uploaded file
    except SlackApiError as e:
        # You will get a SlackApiError if "ok" is False
        assert e.response["ok"] is False
        assert e.response["error"]  # str like 'invalid_auth', 'channel_not_found'
        print(f"Got an error: {e.response['error']}")

    requests.post(response_url,data=json.dumps(payload))
    
    
@app.route('/vis-trigger1', methods=['POST'])
def vis_trigger():
    data = request.form
    #print(data)
    user_id = data.get('user_id')
    channel_id = data.get('channel_id')
    text = data.get('text')
    response_url = data.get("response_url")
    #event = payload.get('event', {})
    #text = event.get('text')
    greeting_message = "Processing your request. Please wait."
    ending_message = "Process executed successfully"

    #utilizing threading
    thr = Thread(target=backgroundworker, args=[text,response_url])
    thr.start()

    return f'{greeting_message}', 200

@app.route('/slack/interactive-endpoint', methods=['GET','POST'])
def interactive_trigger():
    data = request.form
    #print(data)
    #this is the one that I am making use of
    data2 = request.form.to_dict()
    print(data2)
    #data3 = request.args.to_dict()
    #print(data3)
    user_id = data.get('user_id')
    channel_id = data.get('channel_id')
    text = data.get('text')
    response_url = data.get("response_url")
    actions = data.get("actions")
    actions_value = data.get("actions.value")
    #event = payload.get('event', {})
    #text = event.get('text')
    greeting_message = "Processing your request. Please wait."
    ending_message = "Process executed successfully"
    #payload = request.payload
    #payload = json.loads(data2['payload'])
    #obtain the value inserted in the text prompt
    #kw_value=payload['actions'][0]['value']
    
    
    
    if (json.loads(data2['payload'])['actions'][0]['type'] == 'plain_text_input'):
        #utilizing threading
        payload = json.loads(data2['payload'])
        #obtain the value inserted in the text prompt
        kw_value=payload['actions'][0]['value']
        
        #datetime picker block for startdate that is triggered after text input block
        blocks11 = [
            
    		{
    			"type": "input",
    			"element": {
    				"type": "datepicker",
    				"initial_date": "2022-01-01",
    				"placeholder": {
    					"type": "plain_text",
    					"text": "Select a date",
    					"emoji": True
    				},
    				"action_id": "datepicker-action"
    			},
    			"label": {
    				"type": "plain_text",
    				"text": "You may modify the startdate for the Visualization",
    				"emoji": True
    			}
    		}
    	]
        #triggering backgroundworker1 
        thr = Thread(target=backgroundworker, args=[kw_value, response_url])
        thr.start()
        
        #posts in client chat with block 11 as defined above
        client.chat_postMessage(channel='#asb_dd_top10_changes', 
                                                text=f"{kw_value}  ",
                                                blocks=blocks11
                                                )
    elif (json.loads(data2['payload'])['message']['blocks'][0]['label']['text'] == 'You may modify the startdate for the Visualization'):
        kw_value2 = json.loads(data2['payload'])['message']['text'].split()[0]
        init_date = json.loads(data2['payload'])['actions'][0]['selected_date']
        
        
        #datetime picker block for indexdate that is triggered after text input block
        blocks12 = [
            
    		{
    			"type": "input",
    			"element": {
    				"type": "datepicker",
    				"initial_date": "2022-01-01",
    				"placeholder": {
    					"type": "plain_text",
    					"text": "Select a date",
    					"emoji": True
    				},
    				"action_id": "datepicker-action"
    			},
    			"label": {
    				"type": "plain_text",
    				"text": "Pick index date for Visualization",
    				"emoji": True
    			}
    		}
    	]
        
        
        #starts backgroundworker2 function with 3 parameters
        thr = Thread(target=backgroundworker2, args=[kw_value2, init_date, response_url])
        thr.start()
        # if (json.loads(data2['payload'])['actions'][0]['type'] == 'plain_text_input')
        # kw_value = json.loads(data2['payload'])['message']['text'].split()[0]
        # init_date = json.loads(data2['payload'])['actions'][0]['selected_date']
        # thr = Thread(target=backgroundworker2, args=[kw_value, init_date, response_url])
        # thr.start()
        

        #sending kw_value2 and init_date so that we can obtain the values in the elif statement
        client.chat_postMessage(channel='#asb_dd_top10_changes', 
                                                text=f"{kw_value2} {init_date}",
                                                blocks=blocks12
                                                )
    elif (json.loads(data2['payload'])['message']['blocks'][0]['label']['text'] == 'Pick index date for Visualization'):
        kw_value3 = json.loads(data2['payload'])['message']['text'].split()[0]
        init_date2 = json.loads(data2['payload'])['message']['text'].split()[1]
        index_date = json.loads(data2['payload'])['actions'][0]['selected_date']
        
        
        thr = Thread(target=backgroundworker3, args=[kw_value3, init_date2, index_date, response_url])
        thr.start()
        # client.chat_postMessage(channel='#asb_dd_top10_changes', 
        #                                         text=f"{kw_value}  "
        #                                         )    
    else:
        pass
        
        
    
    #this block is released everytime an interactive trigger is activated through the blocks
    fin_block1 = [
		{
			"type": "section",
			"text": {
				"type": "mrkdwn",
				"text": "Please wait for the image to be loaded..."
			},
		},
		{
			"type": "divider"
		},
		{
			"type": "context",
			"elements": [
				{
					"type": "image",
					"image_url": "https://freepngimg.com/thumb/wrench/5-2-wrench-free-download-png.png",
					"alt_text": "update"
				},
				{
					"type": "mrkdwn",
					"text": "You can *modify* has startdate of the visualization and obtain an updated visualization."
				}
			]
		},
		{
			"type": "divider"
		},
		{
			"type": "section",
			"text": {
				"type": "mrkdwn",
				"text": "You can also *change* :chart_with_upwards_trend: has indexdate of the visualization and obtain visualization with updated index."
			}
		}
	]
    
    #obtains the string for file id that is posted last by the bot
    file_id = str(client.files_list(user='U03T83EUHHS',count=1,types='images')['files'][0]['id'])
    #deletes the file with file_id (last file to be posted by the bot)
    client.files_delete(file = file_id)
    
    #this is publishing payload values that needs to be parsed on slack chat
    response2 = client.chat_postMessage(channel='#asb_dd_top10_changes', 
                                        text=" ",
                                        blocks=fin_block1
                                        )
        

    return f'{response2}', 200

#not doing anything at the moment
@app.route("/slack/interactive-endpoint/plain_text_input-action")
def approve_request():
    # Acknowledge action request
    client.chat_postMessage(channel='#asb_dd_top10_changes', 
                                        text="Visualization90:  "
                                        )
    


#testing out blocks in slack
@app.route('/deck-trigger1', methods=['GET', 'POST'])
def deck_trigger():
    data = request.form
    #print(data)
    user_id = data.get('user_id')
    channel_id = data.get('channel_id')
    text = data.get('text')
    response_url = data.get("response_url")
    actions = data.get("actions")
    actions_value = data.get("actions.value")
    #event = payload.get('event', {})
    #text = event.get('text')
    greeting_message = f"Processing your request. Please wait. {actions}"
    ending_message = "Process executed successfully"
    
    
    
    blocks5 = [
		{
			"type": "actions",
			"block_id": "actions1",
			"elements": [
				{
					"type": "static_select",
					"placeholder": {
						"type": "plain_text",
						"text": "Select the type of Vizualization?"
					},
					"action_id": "select_2",
					"options": [
						{
							"text": {
								"type": "plain_text",
								"text": "Brand Deck"
							},
							"value": "first_viz"
						},
						{
							"text": {
								"type": "plain_text",
								"text": "Scatterplot"
							},
							"value": "second_viz"
						}
					]
				},
				{
					"type": "button",
					"text": {
						"type": "plain_text",
						"text": "Cancel"
					},
					"value": "cancel",
					"action_id": "button_1"
				}
			]
		}
	]
    
    blocks6 = [
	{
		"dispatch_action": True,
		"type": "input",
		"element": {
			"type": "plain_text_input",
			"action_id": "plain_text_input-action"
		},
		"label": {
			"type": "plain_text",
			"text": "Please type the keyword for the visualization ",
			"emoji": True
		}
	}
]

    blocks7 = [
		{
			"type": "actions",
			"elements": [
				{
					"type": "radio_buttons",
					"options": [
						{
							"text": {
								"type": "plain_text",
								"text": "Devices Demand",
								"emoji": True
							},
							"value": "value-0"
						},
						{
							"text": {
								"type": "plain_text",
								"text": "Digital Demand",
								"emoji": True
							},
							"value": "value-1"
						}
					],
					"action_id": "actionId-0"
				}
			]
		},
		{
			"type": "input",
			"element": {
				"type": "multi_static_select",
				"placeholder": {
					"type": "plain_text",
					"text": "Select options",
					"emoji": True
				},
				"options": [
					{
						"text": {
							"type": "plain_text",
							"text": "Iphone 13",
							"emoji": True
						},
						"value": "value-0"
					},
					{
						"text": {
							"type": "plain_text",
							"text": "Galaxy S22+",
							"emoji": True
						},
						"value": "value-1"
					},
					{
						"text": {
							"type": "plain_text",
							"text": "Iphone 14 Pro",
							"emoji": True
						},
						"value": "value-2"
					}
				],
				"action_id": "multi_static_select-action"
			},
			"label": {
				"type": "plain_text",
				"text": "Devices",
				"emoji": True
			}
		},
		{
			"type": "actions",
			"elements": [
				{
					"type": "datepicker",
					"initial_date": "1990-04-28",
					"placeholder": {
						"type": "plain_text",
						"text": "Select a date",
						"emoji": True
					},
					"action_id": "actionId-0"
				},
				{
					"type": "datepicker",
					"initial_date": "1990-04-28",
					"placeholder": {
						"type": "plain_text",
						"text": "Select a date",
						"emoji": True
					},
					"action_id": "actionId-1"
				}
			]
		},
		{
			"type": "input",
			"element": {
				"type": "radio_buttons",
				"options": [
					{
						"text": {
							"type": "plain_text",
							"text": "Time-series",
							"emoji": True
						},
						"value": "value-0"
					}
				],
				"action_id": "radio_buttons-action"
			},
			"label": {
				"type": "plain_text",
				"text": "Report Preference",
				"emoji": True
			}
		}
	]
    
    blocks8 = [
		{
			"type": "input",
			"element": {
				"type": "static_select",
				"placeholder": {
					"type": "plain_text",
					"text": "Select an item",
					"emoji": True
				},
				"options": [
					{
						"text": {
							"type": "plain_text",
							"text": "*Viz Trigger*",
							"emoji": True
						},
						"value": "value-0"
					}
				],
				"action_id": "static_select-action"
			},
			"label": {
				"type": "plain_text",
				"text": "Request",
				"emoji": True
			}
		},
		{
			"type": "input",
			"element": {
				"type": "checkboxes",
				"options": [
					{
						"text": {
							"type": "plain_text",
							"text": "Telekom",
							"emoji": True
						},
						"value": "value-0"
					},
					{
						"text": {
							"type": "plain_text",
							"text": "Vodafone",
							"emoji": True
						},
						"value": "value-1"
					},
					{
						"text": {
							"type": "plain_text",
							"text": "o2",
							"emoji": True
						},
						"value": "value-2"
					}
				],
				"action_id": "checkboxes-action"
			},
			"label": {
				"type": "plain_text",
				"text": "Provider",
				"emoji": True
			}
		},
		{
			"type": "input",
			"element": {
				"type": "datepicker",
				"initial_date": "1990-04-28",
				"placeholder": {
					"type": "plain_text",
					"text": "Select a date",
					"emoji": True
				},
				"action_id": "datepicker-action"
			},
			"label": {
				"type": "plain_text",
				"text": "Start Date",
				"emoji": True
			}
		},
		{
			"type": "input",
			"element": {
				"type": "datepicker",
				"initial_date": "1990-04-28",
				"placeholder": {
					"type": "plain_text",
					"text": "Select a date",
					"emoji": True
				},
				"action_id": "datepicker-action"
			},
			"label": {
				"type": "plain_text",
				"text": "End Date",
				"emoji": True
			}
		},
		{
			"type": "input",
			"element": {
				"type": "radio_buttons",
				"options": [
					{
						"text": {
							"type": "plain_text",
							"text": "Datapoints with value",
							"emoji": True
						},
						"value": "value-0"
					},
					{
						"text": {
							"type": "plain_text",
							"text": "Trend Analysis",
							"emoji": True
						},
						"value": "value-1"
					},
					{
						"text": {
							"type": "plain_text",
							"text": "Moving Average",
							"emoji": True
						},
						"value": "value-2"
					}
				],
				"action_id": "radio_buttons-action"
			},
			"label": {
				"type": "plain_text",
				"text": "Label",
				"emoji": True
			}
		},
		{
			"type": "actions",
			"elements": [
				{
					"type": "button",
					"text": {
						"type": "plain_text",
						"text": "Generate",
						"emoji": True
					},
					"value": "click_me_123",
					"action_id": "actionId-0"
				}
			]
		}
	]
    #{blocks4[0]['elements'][1]['value']}
    
# Your listener will be called every time a block element with the action_id "approve_button" is triggered
# @app.action("select_2")
# def update_message(ack):
#     ack()
    
    #this is triggering the slash-cmd called brand_deck1
    response = client.chat_postMessage(channel='#asb_dd_top10_changes', 
                                        text="Visualization:  ",
                                        blocks= blocks6
                                        )
    
    
    
    # payload2 = 
    
    #this code for blocks is working
    # try:
    #     response = client.chat_postMessage(channel='#asb_dd_top10_changes', 
    #                                         text="Visualization: ",
    #                                         blocks= [{"type": "section",
    #                                                   "text": {
    #                                                       "type": "mrkdwn",
    #                                                       "text": "This is a section block with a button."},
    #                                                   "accessory": {"type": "button",
    #                                                                 "type": "plain_text",
    #                                                                 "text": "Click Me",
    #                                                                 "emoji": True},
    #                                                   "value": "click_me_123",
    #                                                   "action_id": "button-action"}
    #                                                  ]
    #                                         )
    #     assert response["message"]["text"] == "Visualization: "
    # except SlackApiError as e:
    #     # You will get a SlackApiError if "ok" is False
    #     assert e.response["ok"] is False
    #     assert e.response["error"]  # str like 'invalid_auth', 'channel_not_found'
    #     print(f"Got an error: {e.response['error']}")
    
    
    #interactive_trigger()
    
    return f"{response_url}", 200

#testing out blocks in slack
@app.route('/radio-trigger1', methods=['GET', 'POST'])
def radio_trigger():
    data = request.form
    #print(data)
    user_id = data.get('user_id')
    channel_id = data.get('channel_id')
    text = data.get('text')
    response_url = data.get("response_url")
    
    

    
    blocks9 = [
		{
			"type": "input",
			"element": {
				"type": "radio_buttons",
				"options": [
					{
						"text": {
							"type": "plain_text",
							"text": "Publish image",
							"emoji": True
						},
						"value": "value-0"
					},
					{
						"text": {
							"type": "plain_text",
							"text": "Print",
							"emoji": True
						},
						"value": "value-1"
					}
				],
				"action_id": "radio_buttons-action"
			},
			"label": {
				"type": "plain_text",
				"text": "Choose Option",
				"emoji": True
			}
		}
	]

    #for using keyword typed in block as a trigger to generate visualizations
    # def vis_trigger2():
    #     data = request.form
    #     data2 = request.form.to_dict()
    #     #print(data)
    #     user_id = data.get('user_id')
    #     channel_id = data.get('channel_id')
    #     text = data.get('text')
    #     response_url = data.get("response_url")
    #     #event = payload.get('event', {})
    #     #text = event.get('text')
    #     greeting_message = "Processing your request. Please wait."
    #     ending_message = "Process executed successfully"
        
    #     #using json.loads to obtain the value of the payload 
    #     #data2 is the payload in dictionary format
    #     payload = json.loads(data2['payload'])
    #     #obtain the value inserted in the text prompt
    #     kw_value=payload['actions'][0]['value']
        
    #     #utilizing threading
    #     thr = Thread(target=backgroundworker, args=[kw_value,response_url])
    #     thr.start()

    #     return f'{greeting_message}', 200

    # vis_trigger2()
    
    #{blocks4[0]['elements'][1]['value']}
    
# Your listener will be called every time a block element with the action_id "approve_button" is triggered
# @app.action("select_2")
# def update_message(ack):
#     ack()
        

    #this is triggering the slash-cmd called radio_trigger1
    response_radio = client.chat_postMessage(channel='#asb_dd_top10_changes', 
                                        text="Visualization:  ",
                                        blocks = blocks9
                                        )
    
    
    
    # payload2 = 
    
    #this code for blocks is working
    # try:
    #     response = client.chat_postMessage(channel='#asb_dd_top10_changes', 
    #                                         text="Visualization: ",
    #                                         blocks= [{"type": "section",
    #                                                   "text": {
    #                                                       "type": "mrkdwn",
    #                                                       "text": "This is a section block with a button."},
    #                                                   "accessory": {"type": "button",
    #                                                                 "type": "plain_text",
    #                                                                 "text": "Click Me",
    #                                                                 "emoji": True},
    #                                                   "value": "click_me_123",
    #                                                   "action_id": "button-action"}
    #                                                  ]
    #                                         )
    #     assert response["message"]["text"] == "Visualization: "
    # except SlackApiError as e:
    #     # You will get a SlackApiError if "ok" is False
    #     assert e.response["ok"] is False
    #     assert e.response["error"]  # str like 'invalid_auth', 'channel_not_found'
    #     print(f"Got an error: {e.response['error']}")
    
    
    #interactive_trigger()
    
    return f"{response_radio}", 200

#calendar trigger slash command
@app.route('/calendar_button1', methods=['POST'])
def calendar_trigger():
    data = request.form
    data2 = request.form.to_dict()
    #print(data)
    user_id = data.get('user_id')
    channel_id = data.get('channel_id')
    text = data.get('text')
    response_url = data.get("response_url")
    #event = payload.get('event', {})
    #text = event.get('text')
    greeting_message = "Processing your request. Please wait."
    ending_message = "Process executed successfully"

    #utilizing threading
    #thr = Thread(target=backgroundworker, args=[text,response_url])
    #thr.start()
    
    blocks11 = [
		{
			"type": "input",
			"element": {
				"type": "datepicker",
				"initial_date": "2022-01-01",
				"placeholder": {
					"type": "plain_text",
					"text": "Select a date",
					"emoji": True
				},
				"action_id": "datepicker-action"
			},
			"label": {
				"type": "plain_text",
				"text": "Label",
				"emoji": True
			}
		}
	]
    

    response_calendar = client.chat_postMessage(channel='#asb_dd_top10_changes', 
                                        text=f"Visualization:  ",
                                        blocks = blocks11
                                        )
    
    return f'{data2}', 200

#files_list trigger slash command
@app.route('/files_list1', methods=['POST'])
def files_trigger():
    data = request.form
    data2 = request.form.to_dict()
    #print(data)
    user_id = data.get('user_id')
    channel_id = data.get('channel_id')
    text = data.get('text')
    response_url = data.get("response_url")
    #event = payload.get('event', {})
    #text = event.get('text')
    greeting_message = "Processing your request. Please wait."
    ending_message = "Process executed successfully"

    #utilizing threading
    #thr = Thread(target=backgroundworker, args=[text,response_url])
    #thr.start()
    
    #obtains last sent file by the bot that are images

    # response_calendar = client.chat_postMessage(channel='#asb_dd_top10_changes', 
    #                                     text=f"Visualization:  ")
    

    #obtains the string for file id that is posted last by the bot
    file_id = str(client.files_list(user='U03T83EUHHS',count=1,types='images')['files'][0]['id'])
    #deletes the file with file_id (last file to be posted by the bot)
    client.files_delete(file = file_id)
    
    client.chat_postMessage(channel='#asb_dd_top10_changes', 
                                            text="test_block"
                                            )
    
    return f'{file_id}', 200

@app.route('/combine_trigger1', methods=['POST'])
def combine_trigger():
    data = request.form
    #we are usging data2 to parse the information
    data2 = request.form.to_dict()
    #print(data)
    user_id = data.get('user_id')
    channel_id = data.get('channel_id')
    text = data.get('text')
    response_url = data.get("response_url")
    #event = payload.get('event', {})
    #text = event.get('text')
    greeting_message = "Processing your request. Please wait."
    ending_message = "Process executed successfully"

    #utilizing threading
    #thr = Thread(target=backgroundworker, args=[text,response_url])
    #thr.start()
    
    #this creates the text prompt in slack block kit
    blocks6 = [
		{
           "type": "divider"
           },
    	{
    		"dispatch_action": True,
    		"type": "input",
    		"element": {
    			"type": "plain_text_input",
    			"action_id": "plain_text_input-action"
    		},
    		"label": {
    			"type": "plain_text",
    			"text": "Please type the keyword for the visualization ",
    			"emoji": True
    		}
    	}
    ]
    
    client.chat_postMessage(channel='#asb_dd_top10_changes', 
                                        text="Visualization:  ",
                                        blocks = blocks6
                                        )
    

    
    #returning empty string with 200 response
    return '', 200


#files_list trigger slash command
@app.route('/hello-world', methods=['POST'])
def hello_world_funct():
        
    client.chat_postMessage(channel='#asb_dd_top10_changes', 
                                            text="hello world"
                                            )
    
    return '', 200


# blocks= [{"type": "section", 
#           "text": {"type": "plain_text", 
#                    "text": "Hello world"}}]

if __name__ == "__main__":
    app.run(debug=True) #debug = True, makes sure if we modify this file we don't need to rerun the python script
    
    
