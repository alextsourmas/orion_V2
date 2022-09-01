from abc import get_cache_token
from tokenize import Double
import numpy as np
import pandas as pd
import math 
import xgboost as xgb
import matplotlib.pyplot as plt
import yfinance as yf
import ta 
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import schedule
import time
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus
from alpaca.trading.requests import GetOrdersRequest
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoLatestQuoteRequest
import os 


import warnings
warnings.filterwarnings("ignore")

alpaca_api_key = os.environ['ALPACA_KEY']
alpaca_api_secret = os.environ['ALPACA_SECRET']

trading_client = TradingClient(alpaca_api_key, alpaca_api_secret, paper=True)


def get_yahoo_stock_data(stock: str, period: str, verbose=True):
    '''
    Get data from yfinance API
    Args: 
        stock(str): ticker name according to yahoo 
        period(str): Examples include 1M, 2M, 3M, 1Y, 2Y, 3Y, 10Y, max
        its not clear what strings to pass in - try it and see if it works
        verbose(bool): choose whether to add pritn statements
    Rtypes: 
        stock_datafram(pd.DataFrame) return dataframe of stock data according to time period 
    '''
    if verbose: print('\nGetting stock data for {}...'.format(stock))
    stock_dataframe = yf.Ticker(stock)
    stock_dataframe = stock_dataframe.history(period=period)
    # stock_dataframe = stock_dataframe.history(period='max')
    stock_dataframe = stock_dataframe.reset_index()
    stock_dataframe['Date'] = stock_dataframe['Date'].astype(str)
    stock_dataframe = stock_dataframe.reset_index(drop=True)
    # stock_dataframe['Ticker'] = stock
    if verbose: print('Loaded stock data into memory.')
    return stock_dataframe 




def time_series_split(stock_df: pd.DataFrame, train_days: int, test_days: int, holdout_days: int, verbose=True): 
    '''
    Rolls through the whole dataset and divides the data into joined (all combined), train, test, and holdout sets
    This is used to simulate backtesting and is a very useful function for any trading algorithm 
    Args: 
    Rtypes: 

    '''

    stock_df = stock_df.reset_index(drop=True) #Very important to reset the index - this function bases time on it 

    #Set how many rounds to run of splitting the data into training, testing, holdout, and "joined" sets
    total_days = len(stock_df)
    rounds = math.ceil((total_days - train_days - test_days) / holdout_days)

    #Empty dictionaries to store datasets in 
    train_df_dict = {}
    test_df_dict = {}
    holdout_df_dict = {}
    joined_df_dict = {}

    #Rolling start value for the dataframes as we roll through the data - this moves every iteration 
    moving_start = 0

    for round in range(0, rounds): #For every round, split the data into respective parts and store in dictionaries 

        train_df = stock_df[moving_start : moving_start + train_days]
        test_df = stock_df[moving_start + train_days : moving_start + train_days + test_days]
        holdout_df = stock_df[ moving_start + train_days + test_days:  moving_start + train_days + test_days + holdout_days]
        joined_df = pd.concat([train_df, test_df, holdout_df])
        
        train_df_dict[round] = train_df
        test_df_dict[round] = test_df
        holdout_df_dict[round] = holdout_df
        joined_df_dict[round] = joined_df

        moving_start = moving_start + holdout_days #Move the moving start value after every round 

    return joined_df_dict, train_df_dict, test_df_dict, holdout_df_dict 





def simple_moving_average(stock_df: pd.DataFrame, close_col: str, window: int, fillna: True, verbose = True): 
    '''
    Generate a simple moving average on a dataframe - save to a new column 
    Args: 
        stock_df(pd.DataFrame): dataframe to use
        close_col(str): name of the close column 
        window(int): lookback window for the moving average
        fillna(bool): choose whether to fill na's (especially if the df isn't big enough for the lookback...itll
        use as many rows as are available instead)
        verbose(bool): choose whether to add prints to the function 
    Rtypes:
        rolling_series(pd.Series): Series with the rolling average
    '''
    if verbose: print('\nGenerating simple moving average for {} days...'.format(window))
    min_periods = 0 if fillna else window 
    rolling_series = stock_df[close_col].rolling(window=window, min_periods=min_periods).mean()
    if verbose: print('SMA calcualted for time period {}.'.format(window))
    return rolling_series




def trend_analysis(stock_df: pd.DataFrame, window: int, sma_col: str, close_col: str, verbose=True): 
    '''
    Perform trend analysis 
    If closing price value leads its MA15 and MA15 is rising for last n days then trend is Uptrend i.e. trend signal is 1.
    If closing price value lags its MA15 and MA15 is falling for last n days then trend is Downtrend i.e. trend signal is 0.
    
    Args: 
        stock_df(pd.DataFrame) dataframe with stock info and moving average
        window(int): how long to use for a window for the trend analysis 
        sma_col(str): name of 15 day moving average col to use (can use a different one)
        close_col(str): name of the close column
        verbose(bool): add print statements
    Rtypes: 
        series(pd.Series): column with the new trend analysis in it 
    '''

    if verbose: print('\nGetting trend analysis...')
    stock_df = stock_df.reset_index(drop=True)
    stock_df['trend_analysis'] = ''
    for row in range(0, len(stock_df)): 
        #Set variables 
        current_close = stock_df[close_col].loc[row]
        current_ma = stock_df[sma_col].loc[row]
        slice_df = stock_df.loc[row - (window - 1): row]
        monotonic_increasing = slice_df[sma_col].is_monotonic_increasing
        monotonic_decreasing = slice_df[sma_col].is_monotonic_decreasing
        #If conditions are met, set trend
        if (current_close > current_ma) and monotonic_increasing: 
            trend = 'up'
        elif (current_close < current_ma) and monotonic_decreasing: 
            trend = 'down' 
        else:
            trend = 'no'
        
        stock_df['trend_analysis'].loc[row] = trend

    series = stock_df['trend_analysis']
    stock_df = stock_df.drop(columns='trend_analysis', inplace=True)
    if verbose: print('Trend analysis complete.')
    return series 




def get_quantified_trend_backward_looking(stock_df: pd.DataFrame, close_col: str, trend_analysis_col: str, window=3, verbose=True):
    '''
    Quantify the trend variables based on the equation provided
    Args: 
        stock_df(pd.Dataframe) stock df to use
        close_col(str): name of close column
        trend_analysis_col(str): name of trend analysis column
        window(int): window size to use for the analysis (default is 3 according to the paper)
        verbose(bool): choose whether to add prints
    Rtypes: 
        stock_df(pd.DataFrame): finished dataframe with quantified trend column
    '''
    
    if verbose: print('\nCalculating stock quantified trend...')
    stock_df = stock_df.reset_index(drop=True)
    stock_df['quantified_trend'] = ''

    for row in range(0, len(stock_df)):

        slice_df = stock_df.loc[row - (window - 1): row]
        current_trend = stock_df[trend_analysis_col].loc[row]
        current_close = stock_df[close_col].loc[row]
        min_cp = slice_df[close_col].min()
        max_cp = slice_df[close_col].max()
        #Check if denominator is zero - if it is, return zero rather than throw an error
        if (max_cp - min_cp) != 0:
            value_if_uptrend_or_hold = ((current_close - min_cp)/(max_cp - min_cp) * 0.5) + 0.5
            value_if_downtrend = ((current_close - min_cp)/(max_cp - min_cp) * 0.5)
        else: 
            value_if_uptrend_or_hold = 0
            value_if_downtrend = 0
        #Set values based on condition
        if current_trend == 'up': 
            stock_df['quantified_trend'].loc[row] = value_if_uptrend_or_hold

        if current_trend == 'no': 
            stock_df['quantified_trend'].loc[row] = value_if_uptrend_or_hold 

        if current_trend == 'down': 
            stock_df['quantified_trend'].loc[row] = value_if_downtrend

    series = stock_df['quantified_trend']
    stock_df = stock_df.drop(columns='quantified_trend', inplace=True)
    if verbose: print('Calculated stock quantified trend.')
    return series




def generate_trade_signal(stock_df: pd.DataFrame, quantified_trend_col: str, strategy: str, quantile_value: float, static_value = float, verbose=True): 
    '''
    Generate the final trade signal used to create buy and sell decisions - set the buy and sell cutoff based on mean, 
    median, or quantile. 
    Args: 
        stock_df(pd.DataFrame): dataframe of stock data
        quantified_trend_col(str): name of quantified trend column
        strategy(str): mean, median, or quantile to set cutoff threshold between buy and sell (IMPORTANT PARAMETER - the 
        whole model and performance will be determined by this...)
        quantile_value(str): set regardless, only but used if your strategy is quantile_value
        static_value(float): set a static cutoff threshold rather than dynamic 
    Rtypes: 
        series(pd.Series): new column with the final trade signals generated 
    '''
    stock_df = stock_df.reset_index(drop=True)
    if verbose: print('\nGenerating trade signal (Up or Down)...') #Set cutoff method 
    if strategy == 'mean': 
        cutoff_value = stock_df[quantified_trend_col].mean()
    if strategy == 'median': 
        cutoff_value = stock_df[quantified_trend_col].median()
    if strategy == 'quantile': 
        cutoff_value = stock_df[quantified_trend_col].quantile(quantile_value)
    if strategy == 'static': 
        cutoff_value = static_value

    stock_df['trade_signal'] = '' #Create empty col to fill values 

    for row in range(0, len(stock_df)): #If above cutoff, Up, else, down 
        quantified_trend = stock_df[quantified_trend_col].loc[row]
        if quantified_trend > cutoff_value: 
            trend = 'Up'
        else: 
            trend = 'Down'
        stock_df['trade_signal'].loc[row] = trend

    series = stock_df['trade_signal'] #Return trade signal
    stock_df = stock_df.drop(columns='trade_signal', inplace=True)
    if verbose: print('Trade signal generated.')
    return series 


def generate_buy_decision(stock_df: pd.DataFrame, trade_signal_col_name: str, verbose=True): 
    '''
    Generate the buy and sell decisions in the dataframe based on the up and down condition in the trade signals column
    Args: 
        stock_df(pd.Dataframe) df to create buy and sell decisions on 
        trade_signal_col_name(str): name of column with trade signals created
    Rtypes: 
        series(pd.Series): column with the buy and sell decisions in it 
    '''
    stock_df = stock_df.reset_index(drop=True)
    if verbose: print('\nGenerating buy or sell decisions...')
    stock_df['buy_decision'] = ''

    for row in range(0, len(stock_df)): 
        
        current_condition = stock_df[trade_signal_col_name].iloc[row]
        prior_condition = stock_df[trade_signal_col_name].iloc[row-1]

        if (current_condition != prior_condition) & (prior_condition == 'Up'):
            final_decision = 'sell'
        elif (current_condition != prior_condition) & (prior_condition == 'Down'):
            final_decision = 'buy'
        else: 
            final_decision = 'hold'
        
        stock_df['buy_decision'].iloc[row] = final_decision

    series = stock_df['buy_decision']
    stock_df = stock_df.drop(columns=['buy_decision'], inplace=True)
    if verbose: print('Buy or sell decisions generated.')
    return series 




def rolling_predictor_auto_tuner(joined_df_dict: dict, train_df_dict: dict, test_df_dict: dict, holdout_df_dict: dict,\
    sma_window_list: list, trend_analysis_window_list: list, quantified_trend_window_list: list,\
    cutoff_threshold_list: list, random_state = 1, verbose=True, ticker: str='BTC-USD'):
    '''
    Automatically rolls through the data, training, testing, and predicting on holdout sets
    If you use the input lists you can set it to optimize alpha on every test period and use those...
    settings on the holdout period after. "Auto-tuner" 
    Args: 
        joined_df_dict(dict): dict of the joined data
        train_df_dict(dict): dictionary of every training set
        test_df_dict(dict): dictionary of every test set
        holdout_df_dict(dict): dictionary of every holdout set
        sma_window_list(list): first variable to tweak in the algorithm, changes SMA
        trend_analysis_window_list(list): second variable to tweak, changes trend analysis
        quantified_trend_window_list(list): third variable to tweak, changes trend quantification
        cutoff_threshold_list(list): fourth variable to tweak, changes cutoff between buy and sell
        random_state(int): random state for the XGB model - keep this constant 
        verbose(bool): Choose whether to add prints
    Rtypes: 
        final_holdout_results_df(pd.Dataframe): df of all the holdout sets joined together 
    '''

    current_iteration = 0

    for dataset in range(0, len(holdout_df_dict)): 
        
        train_df = train_df_dict[dataset]
        test_df = test_df_dict[dataset]
        holdout_df = holdout_df_dict[dataset]
        joined_df = joined_df_dict[dataset]
        tuning_df = pd.DataFrame(columns=['ticker', 'sma_window', 'trend_analysis_window', 'quantified_trend_window', 'cutoff', 'starting_asset_value',\
            'ending_asset_value', 'baseline_p_and_l', 'starting_portfolio_value', 'ending_portfolio_value', 'strategy_p_and_l', 'alpha', 'num_trades', 'sharpe_ratio'])
        total_iterations = len(sma_window_list) * len(trend_analysis_window_list) * len(quantified_trend_window_list) * len(cutoff_threshold_list) * len(holdout_df_dict)

        for sma_window_counter in range(0, len(sma_window_list)): #For every value in the tuner 
            sma_window = sma_window_list[sma_window_counter]
            for trend_analysis_window_counter in range(0, len(trend_analysis_window_list)): 
                trend_analysis_window = trend_analysis_window_list[trend_analysis_window_counter]
                for quantified_trend_counter in range(0, len(quantified_trend_window_list)): 
                    quantified_trend_window = quantified_trend_window_list[quantified_trend_counter]
                    for cutoff_counter in range(0, len(cutoff_threshold_list)): 
                        cutoff = cutoff_threshold_list[cutoff_counter]


                        #Joined df feature engineering and manipulation  
                        joined_df['index_col'] = joined_df.index
                        joined_df = joined_df.reset_index(drop=True)
                        joined_df['sma'] = simple_moving_average(stock_df = joined_df, close_col='Close',\
                            window=sma_window, fillna=True, verbose=False)
                        joined_df['trend_analysis'] = trend_analysis(stock_df= joined_df, window= trend_analysis_window, sma_col= 'sma',\
                            close_col= 'Close', verbose=False)
                        joined_df['quantified_trend'] = get_quantified_trend_backward_looking(stock_df= joined_df, close_col= 'Close',\
                            trend_analysis_col= 'trend_analysis', window=quantified_trend_window, verbose=False)
                        joined_df = joined_df.set_index('index_col')

                        #trained and test df splitting, splitting for train and test, etc. 
                        temp_train_df = joined_df.loc[train_df.index, :]
                        temp_test_df = joined_df.loc[test_df.index, :]
                        temp_train_df = temp_train_df.reset_index(drop=True)
                        temp_test_df = temp_test_df.reset_index(drop=True)
                        train_dates_df = temp_train_df[['Date']]
                        x_train = temp_train_df.drop(columns=['Date', 'trend_analysis', 'quantified_trend'])
                        y_train = temp_train_df[['quantified_trend']]
                        test_dates_df = temp_test_df[['Date']]
                        x_test = temp_test_df.drop(columns=['Date', 'trend_analysis', 'quantified_trend'])
                        y_test = temp_test_df[['quantified_trend']]

                        if verbose: 
                            print('\nSMA Window: {}'.format(str(sma_window)))
                            print('Trend Analysis Window: {}'.format(str(trend_analysis_window)))
                            print('Quantified Trend Window: {}'.format(str(quantified_trend_window)))
                            print('Cutoff threshold: {}'.format(str(cutoff)))

                        #Train the model 
                        if verbose: print('\nTraining XGBoost model...')
                        xgbr = xgb.XGBRegressor(random_state = random_state)
                        xgbr.fit(x_train, y_train['quantified_trend'])   

                        #Get predictions, calculate some metrics, return predictions dataframe
                        if verbose: print('\nGetting predictions on test set...')
                        y_pred = xgbr.predict(x_test)
                        if verbose: print('PERFORMANCE:')
                        if verbose: print('R2 Score: {}'.format(r2_score(y_test['quantified_trend'], y_pred)))
                        if verbose: print('Mean Squared Error: {}'.format(mean_squared_error(y_test['quantified_trend'], y_pred)))
                        y_pred_df = pd.DataFrame(y_pred, columns = ['predictions'])

                        #Engineer final test set, calculate profit and loss
                        if verbose: print('\nJoining data back together...')
                        test_dates_df = test_dates_df.reset_index(drop=True)
                        x_test = x_test.reset_index(drop=True)
                        y_test = y_test.reset_index(drop=True)
                        y_pred_df = y_pred_df.reset_index(drop=True)
                        final_test_set = pd.concat([test_dates_df, x_test, y_test, y_pred_df], axis=1).reset_index(drop=True)
                        final_test_set['trade_signal'] = generate_trade_signal(stock_df = final_test_set, quantified_trend_col = 'predictions', strategy= 'static', quantile_value= 1, static_value = cutoff, verbose=False) 
                        final_test_set['buy_decision'] = generate_buy_decision(stock_df = final_test_set, trade_signal_col_name= 'trade_signal', verbose=False)
                        earnings_df = final_test_set[['Date', 'Close', 'quantified_trend', 'predictions', 'trade_signal', 'buy_decision']]
                        earnings_df = calculate_profit_and_loss(stock_df= earnings_df,  close_col= 'Close', buy_decision_col= 'buy_decision', initial_cash=100000, verbose=True, commission_percent=.0060)

                        #Calculate profit and loss 
                        starting_asset_value = earnings_df['Close'].loc[0]
                        ending_asset_value = earnings_df['Close'].loc[len(earnings_df) - 1]
                        baseline_p_and_l =  ((ending_asset_value - starting_asset_value) / starting_asset_value) * 100
                        starting_portfolio_value = earnings_df['total_portfolio_value'].loc[0]
                        ending_portfolio_value = earnings_df['total_portfolio_value'].loc[len(earnings_df) - 1]
                        strategy_p_and_l = ((ending_portfolio_value - starting_portfolio_value) / starting_portfolio_value) * 100
                        alpha = strategy_p_and_l - baseline_p_and_l
                        num_trades = len(earnings_df[earnings_df['buy_decision'] != 'hold' ])
                        sharpe_ratio = alpha / earnings_df['total_portfolio_value'].std()

                        #Append results to tuning df 
                        temp_df = {'ticker': ticker, 'sma_window': sma_window, 'trend_analysis_window': trend_analysis_window, 'quantified_trend_window': quantified_trend_window,\
                            'cutoff': cutoff, 'starting_asset_value': starting_asset_value, 'ending_asset_value': ending_asset_value, 'baseline_p_and_l': baseline_p_and_l,\
                                'starting_portfolio_value': starting_portfolio_value, 'ending_portfolio_value': ending_portfolio_value, 'strategy_p_and_l': strategy_p_and_l,\
                                    'alpha': alpha, 'num_trades': num_trades, 'sharpe_ratio': sharpe_ratio}   
                        tuning_df = tuning_df.append(temp_df, ignore_index=True)

                        current_iteration = current_iteration + 1
                        if verbose: print('\nFinished round: ' + str(current_iteration) + ' of ' + str(total_iterations))

        #Get the best performing settings
        tuning_df = tuning_df.sort_values(by=['alpha','num_trades'], ascending=False).reset_index(drop=True)
        sma_window = tuning_df['sma_window'].loc[0]
        trend_analysis_window = tuning_df['trend_analysis_window'].loc[0]
        quantified_trend_window = tuning_df['quantified_trend_window'].loc[0]
        cutoff = tuning_df['cutoff'].loc[0]

        #Instantiate varaibles once again
        train_df = train_df_dict[dataset]
        test_df = test_df_dict[dataset]
        holdout_df = holdout_df_dict[dataset]
        joined_df = joined_df_dict[dataset]

        #Roll through joined data with all the best settings
        joined_df['index_col'] = joined_df.index
        joined_df = joined_df.reset_index(drop=True)
        joined_df['sma'] = simple_moving_average(stock_df = joined_df, close_col='Close',\
            window=sma_window, fillna=True, verbose=False)
        joined_df['trend_analysis'] = trend_analysis(stock_df= joined_df, window= trend_analysis_window, sma_col= 'sma',\
            close_col= 'Close', verbose=False)
        joined_df['quantified_trend'] = get_quantified_trend_backward_looking(stock_df= joined_df, close_col= 'Close',\
            trend_analysis_col= 'trend_analysis', window=quantified_trend_window, verbose=False)
        joined_df = joined_df.set_index('index_col')

        #Split into two datasets, train+test, and holdout 
        train_and_test = joined_df[~joined_df.index.isin(holdout_df.index)]
        holdout_df = joined_df[~joined_df.index.isin(train_and_test.index)]
        train_and_test = train_and_test.reset_index(drop=True)
        holdout_df = holdout_df.reset_index(drop=True)


        #Split out data for training on the model 
        train_and_test_dates_df = train_and_test[['Date']]
        x_train_and_test = train_and_test.drop(columns=['Date', 'trend_analysis', 'quantified_trend'])
        y_train_and_test = train_and_test[['quantified_trend']]
        holdout_dates_df = holdout_df[['Date']]
        x_holdout = holdout_df.drop(columns=['Date', 'trend_analysis', 'quantified_trend'])
        y_holdout = holdout_df[['quantified_trend']]


        #Train XGB on train+test, predict on holdout
        if verbose: print('Training XGBoost model...')
        xgbr = xgb.XGBRegressor(random_state = random_state)
        xgbr.fit(x_train_and_test, y_train_and_test['quantified_trend'])   

        #Predict on the holdout set 
        if verbose: print('\nGetting predictions on holdout set...')
        y_pred = xgbr.predict(x_holdout)
        if verbose: print('PERFORMANCE:')
        if verbose: print('R2 Score: {}'.format(r2_score(y_holdout['quantified_trend'], y_pred)))
        if verbose: print('Mean Squared Error: {}'.format(mean_squared_error(y_holdout['quantified_trend'], y_pred)))
        y_pred_df = pd.DataFrame(y_pred, columns = ['predictions']) 
        if verbose: print('\nJoining data back together...')
        holdout_dates_df = holdout_dates_df.reset_index(drop=True)
        x_holdout = x_holdout.reset_index(drop=True)
        y_holdout = y_holdout.reset_index(drop=True)
        y_pred_df = y_pred_df.reset_index(drop=True)
        final_holdout_df = pd.concat([holdout_dates_df, x_holdout, y_holdout, y_pred_df], axis=1).reset_index(drop=True)

        #Add buy and sell column
        final_holdout_df['trade_signal'] = generate_trade_signal(stock_df = final_holdout_df, quantified_trend_col = 'predictions', strategy= 'static', quantile_value= 1, static_value = cutoff, verbose=False) 
        final_holdout_df['buy_decision'] = generate_buy_decision(stock_df = final_holdout_df, trade_signal_col_name= 'trade_signal', verbose=False)
        # final_holdout_df = calculate_profit_and_loss(stock_df= final_holdout_df,  close_col= 'Close', buy_decision_col= 'buy_decision', initial_cash=100000, verbose=True, commission_percent=.0060)

        if dataset == 0: 
            final_holdout_results_df = final_holdout_df
        else: 
            final_holdout_results_df = final_holdout_results_df.append(final_holdout_df, ignore_index=True)
    
    final_holdout_results_df = final_holdout_results_df.reset_index(drop=True)
    return final_holdout_results_df





def calculate_profit_and_loss(stock_df: pd.DataFrame,  close_col: str, buy_decision_col: str, initial_cash: int, commission_percent: float, verbose=True, ):
    '''
    Make trading decisions, calculate profit and loss for a long only fund 
    Args:
        stock_df(pd.Dataframe): dataframe with  
        close_col(str): name of close column
        buy_decision_col(str): name of buy decision column
        initial_cash(int): starting cash position
        commission_percent(float): add commission to each trade 
        verbose(bool): choose whether to add prints
    Rtypes: 
        stock_df(pd.Dataframe): Dataframe with new profit and loss columns added

    '''

    if verbose: print('\nMaking trading decisions...calculating profit and loss...')
    initial_close_price = stock_df[close_col].loc[0]
    shares = math.floor(initial_cash / initial_close_price)
    share_value = shares * initial_close_price
    leftover_cash = initial_cash - (shares * initial_close_price)

    stock_df['shares_owned'] = ''
    stock_df['total_value_of_shares'] = ''
    stock_df['remaining_cash'] = ''
    stock_df['total_portfolio_value'] = ''

    for i in range(0, len(stock_df)): 
        current_decision = stock_df[buy_decision_col].loc[i]
        todays_close_price = stock_df[close_col].loc[i]
        if (current_decision == 'buy'): 
            if i == 0:
                shares_to_buy = (leftover_cash * (1-commission_percent)) / todays_close_price
                shares = shares + shares_to_buy
                leftover_cash = leftover_cash - (shares_to_buy * todays_close_price)
                stock_df['shares_owned'].loc[i] = shares
                stock_df['total_value_of_shares'].loc[i] = shares * todays_close_price
                stock_df['remaining_cash'].loc[i] = leftover_cash
                stock_df['total_portfolio_value'].loc[i] = (shares * todays_close_price) + leftover_cash          
            else: 
                shares_to_buy =( stock_df['remaining_cash'].loc[i-1] * (1-commission_percent))/ todays_close_price
                shares = stock_df['shares_owned'].loc[i-1] + shares_to_buy
                stock_df['shares_owned'].loc[i] = shares
                stock_df['total_value_of_shares'].loc[i] = shares * todays_close_price
                stock_df['remaining_cash'].loc[i] = stock_df['remaining_cash'].loc[i-1] - (shares_to_buy * todays_close_price)
                stock_df['total_portfolio_value'].loc[i] = stock_df['total_portfolio_value'].loc[i-1]
        if (current_decision == 'hold'): 
                if i == 0:    
                    stock_df['shares_owned'].loc[i] = shares
                    stock_df['remaining_cash'].loc[i] = leftover_cash
                else:  
                    stock_df['shares_owned'].loc[i] = stock_df['shares_owned'].loc[i-1]
                    stock_df['remaining_cash'].loc[i] = stock_df['remaining_cash'].loc[i-1]
                stock_df['total_value_of_shares'].loc[i] = stock_df['shares_owned'].loc[i] * todays_close_price
                stock_df['total_portfolio_value'].loc[i] = stock_df['total_value_of_shares'].loc[i] + stock_df['remaining_cash'].loc[i]
        if (current_decision == 'sell'): 
                if i == 0: 
                    shares_to_sell = shares
                    stock_df['shares_owned'].loc[i] = 0
                    stock_df['total_value_of_shares'].loc[i] = 0 
                    stock_df['remaining_cash'].loc[i] = leftover_cash + ((shares_to_sell * stock_df[close_col].loc[i]) *  (1-commission_percent))
                    stock_df['total_portfolio_value'].loc[i] = stock_df['remaining_cash'].loc[i]
                else: 
                    shares_to_sell = stock_df['shares_owned'].loc[i-1]
                    stock_df['shares_owned'].loc[i] = 0
                    stock_df['total_value_of_shares'].loc[i] = 0 
                    stock_df['remaining_cash'].loc[i] = stock_df['remaining_cash'].loc[i-1] + ((shares_to_sell * stock_df[close_col].loc[i]) * (1-commission_percent))
                    stock_df['total_portfolio_value'].loc[i] = stock_df['remaining_cash'].loc[i]    
    
    if verbose: print('Trading decisions made.')
    starting_asset_value = stock_df[close_col].loc[0]
    ending_asset_value = stock_df[close_col].loc[len(stock_df) - 1]
    baseline_p_and_l =  round(((ending_asset_value - starting_asset_value) / starting_asset_value) * 100, 2)
    starting_portfolio_value = stock_df['total_portfolio_value'].loc[0]
    ending_portfolio_value = stock_df['total_portfolio_value'].loc[len(stock_df) - 1]
    strategy_p_and_l = round(((ending_portfolio_value - starting_portfolio_value) / starting_portfolio_value) * 100, 2)
    alpha = round(strategy_p_and_l - baseline_p_and_l, 2)
    if verbose: print('\nStarting Value Asset: {}'.format(starting_asset_value))
    if verbose: print('Ending Value Asset: {}'.format(ending_asset_value))
    if verbose: print('Baseline P&L: {}%'.format(baseline_p_and_l))
    if verbose: print('\nStarting Portfolio Value: {}'.format(starting_portfolio_value))
    if verbose: print('Ending Portfolio Value: {}'.format(ending_portfolio_value))
    if verbose: print('STRATEGY P&L: {}%'.format(strategy_p_and_l))
    if verbose: print('ALPHA: {}%'.format(alpha))

    return stock_df


def view_portfolio_df(stock_df: pd.DataFrame, total_portfolio_value_col: str, ticker: str):
    '''
    View the portfolio value over time in a graph
    Args: 
        stock_df(pd.DataFrame): Stock df with portfolio value in it
        total_portfolio_value_col(str): Name of column to graph
    Rtypes: 
        None
    '''
    fig, ax = plt.subplots(figsize=(14,8))
    ax.plot(stock_df[total_portfolio_value_col] ,linewidth=0.5, color='blue', alpha = 0.9)
    ax.set_title(ticker,fontsize=10, backgroundcolor='blue', color='white')
    ax.set_ylabel('Porfolio Value' , fontsize=18)
    # legend = ax.legend()
    ax.grid()
    plt.tight_layout()
    plt.show()
    return None



def view_stock_with_decision(stock_df: pd.DataFrame, ticker: str, close_col: str, buy_decision_col: str):
    '''
    View stock dataframe with trading decisions in it 
    Args: 
        stock_dfD(pd.Dataframe): stock df with buy and sell decisions
        ticker(str): ticker name string
        close_col(str): close column name string
        buy_decision_col(str): buy decision column name 
    Rtypes: 
        None
    '''
    def create_buy_column(row): #Create values where decisision is buy 
        if row[buy_decision_col] == 'buy': 
            return row[close_col]
        else: 
            None

    def create_sell_column(row): #Create values where decision is sell
        if row[buy_decision_col] == 'sell': 
            return row[close_col]
        else: 
            return None 
    #Plot all values 
    stock_df['sell_close'] = stock_df.apply(create_sell_column, axis=1)
    stock_df['buy_close'] = stock_df.apply(create_buy_column, axis=1)
    fig, ax = plt.subplots(figsize=(14,8))
    ax.plot(stock_df[close_col] , label = 'Close' ,linewidth=0.5, color='blue', alpha = 0.9)
    ax.scatter(stock_df.index , stock_df['buy_close'] , label = 'Buy' , marker = '^', color = 'green',alpha =1 )
    ax.scatter(stock_df.index , stock_df['sell_close'] , label = 'Sell' , marker = 'v', color = 'red',alpha =1 )
    ax.set_title(ticker + " Price History with Buy and Sell Signals",fontsize=10, backgroundcolor='blue', color='white')
    ax.set_ylabel('Close Prices' , fontsize=18)
    legend = ax.legend()
    ax.grid()
    plt.tight_layout()
    plt.show()
    stock_df = stock_df.drop(columns=['sell_close', 'buy_close'])
    return None










# asset_list = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD', 'SOL-USD', 'DOGE-USD', 'DOT-USD', 'SHIB-USD',\
#     'MATIC-USD', 'AVAX-USD', 'TRX-USD', 'UNI1-USD', 'LTC-USD', 'CRO-USD', 'ATOM-USD', 'XMR-USD', 'XLM-USD']

def build_model_and_get_decision(ticker: str):
    #Set ticker 

    print('\nGetting data for ' + ticker)
    historical_data_df = get_yahoo_stock_data(stock = ticker, period='3Y', verbose= True)
    # historical_data_df = get_stock_data(stock = current_ticker, period='10Y', verbose= False)

    historical_data_df = ta.add_all_ta_features(historical_data_df, open="Open", high="High", low="Low",\
            close = "Close", volume="Volume", fillna=True)

    joined_df_dict, train_df_dict, test_df_dict, holdout_df_dict  = time_series_split(stock_df= historical_data_df, train_days=365, test_days=120, holdout_days=30, verbose=True)

    #Set variables for the auto-tuner and prediction model 
    sma_window_list = [15]
    trend_analysis_window_list = [5]
    quantified_trend_window_list = [3]
    cutoff_threshold_list = [0.87]

    final_holdout_results_df = rolling_predictor_auto_tuner(joined_df_dict= joined_df_dict, train_df_dict= test_df_dict, test_df_dict= test_df_dict,\
        holdout_df_dict= holdout_df_dict, \
        sma_window_list= sma_window_list, trend_analysis_window_list= trend_analysis_window_list, quantified_trend_window_list= quantified_trend_window_list,\
        cutoff_threshold_list= cutoff_threshold_list, random_state = 1, verbose=True, ticker=ticker)


    final_holdout_results_df = calculate_profit_and_loss(stock_df= final_holdout_results_df,  close_col= 'Close', buy_decision_col= 'buy_decision', initial_cash=100000, verbose=True, commission_percent=.0060)

    # view_portfolio_df(stock_df= final_holdout_results_df, total_portfolio_value_col= 'total_portfolio_value', ticker=ticker)

    # view_stock_with_decision(stock_df= final_holdout_results_df, ticker= ticker, close_col= 'Close', buy_decision_col= 'buy_decision')


    #quickly calculate profit but don't save it, just print it 
    starting_asset_value = final_holdout_results_df['Close'].loc[0]
    ending_asset_value = final_holdout_results_df['Close'].loc[len(final_holdout_results_df) - 1]
    baseline_p_and_l =  round(((ending_asset_value - starting_asset_value) / starting_asset_value) * 100, 2)
    starting_portfolio_value = final_holdout_results_df['total_portfolio_value'].loc[0]
    ending_portfolio_value = final_holdout_results_df['total_portfolio_value'].loc[len(final_holdout_results_df) - 1]
    strategy_p_and_l = round(((ending_portfolio_value - starting_portfolio_value) / starting_portfolio_value) * 100, 2)
    alpha = round(strategy_p_and_l - baseline_p_and_l, 2)
    print('\nStarting Value Asset: {}'.format(starting_asset_value))
    print('Ending Value Asset: {}'.format(ending_asset_value))
    print('Baseline P&L: {}%'.format(baseline_p_and_l))
    print('\nStarting Portfolio Value: {}'.format(starting_portfolio_value))
    print('Ending Portfolio Value: {}'.format(ending_portfolio_value))
    print('STRATEGY P&L: {}%'.format(strategy_p_and_l))
    print('ALPHA: {}%'.format(alpha))
    return final_holdout_results_df

def execute_paper_buy_order(ticker: str='BTC/USD', cash: Double=100):
    # no keys required
    quote_client = CryptoHistoricalDataClient()

    # single symbol request
    request_params = CryptoLatestQuoteRequest(symbol_or_symbols=ticker)


    current_ticker_price = quote_client.get_crypto_latest_quote(request_params)[ticker].ask_price

    shares = cash / current_ticker_price

    print('Buying ' + str(shares) + ' shares of ' + ticker)

    # preparing orders
    market_order_data = MarketOrderRequest(
                        symbol=ticker,
                        qty=shares,
                        side=OrderSide.BUY,
                        time_in_force=TimeInForce.GTC
                        )

    # Market order
    market_order = trading_client.submit_order(
                    order_data=market_order_data
                )

    print(market_order)

    # params to filter orders by
    request_params = GetOrdersRequest(
                        status=OrderStatus.FILLED,
                        side=OrderSide.BUY
                    )

    # orders that satisfy params
    orders = trading_client.get_orders(filter=request_params)
    print(orders)

def execute_paper_sell_order(ticker: str = 'BTC/USD', shares: Double = .0001):
    positions = trading_client.get_all_positions()

    shares_to_sell = shares

    for position in positions:
        if (ticker == 'BTC/USD' and position.symbol == 'BTCUSD'):
            shares_to_sell = position.qty

    print('Selling ' + shares_to_sell + ' shares of ' + ticker)

    # preparing orders
    market_order_data = MarketOrderRequest(
                        symbol=ticker,
                        qty=shares_to_sell,
                        side=OrderSide.SELL,
                        time_in_force=TimeInForce.GTC
                        )

    # Market order
    market_order = trading_client.submit_order(
                    order_data=market_order_data
                )

    print(market_order)

    # params to filter orders by
    request_params = GetOrdersRequest(
                        status=OrderStatus.FILLED,
                        side=OrderSide.SELL
                    )

    # orders that satisfy params
    orders = trading_client.get_orders(filter=request_params)
    print(orders)

def get_available_cash():
    return trading_client.get_account()


def job(ticker: str = 'BTC-USD', cash_ratio: Double = 1000):
    resulting_df = build_model_and_get_decision(ticker)
    todays_decision = resulting_df.iloc[-1]
    buy_decision = todays_decision['buy_decision']
    print("Today's Date: " + todays_decision['Date'] + "  Buy Decision: " + buy_decision)

    if (buy_decision == 'buy'):
        cash = get_available_cash()['cash'] / cash_ratio

        execute_paper_buy_order('BTC/USD', cash)
    
    elif (buy_decision == 'sell'):
        execute_paper_sell_order('BTC/USD')

    elif (buy_decision == 'hold'):
        print('The robot says to hold. We stay strong today.')

    return


schedule.every().day.at("19:32").do(job,'BTC-USD',100)

while True:
    schedule.run_pending()
















#Join all these holdout datasets together into one big dataset

#500 training days  @ 0.85 -> 52% alpha
#365 training days  @ 0.85 -> 108% alpha
#200 training days @ 0.85 -> 108% alpha
#120 training days @ 0.85  ->  -69% alpha


#365 training days  @ 0.75 -> 103% alpha
#365 training days  @ 0.80 -> 94% alpha
#365 training days  @ 0.82 -> 84% alpha
#365 training days  @ 0.84 -> 111% alpha
#365 training days  @ 0.85 -> 108% alpha
#365 training days  @ 0.87 -> 108% alpha
#365 training days  @ 0.90 -> 88% alpha
#365 training days  @ 0.95 -> 63% alpha










