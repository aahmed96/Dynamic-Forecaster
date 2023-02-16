"""Library for time-serie framework
"""
import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn import metrics
from sklearn.linear_model import LinearRegression
import holidays
#LSTM stuff
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense
#prophet stuff
from prophet import Prophet
#from prophet.plot import add_changepoints_to_plot
from prophet.utilities import regressor_coefficients

os.environ['TF_DETERMINISTIC_OPS'] = '1'

#helper functions
def evaluation_metrics(y_true, y_pred,print_results=False):
    """Helper function for computing metrics"""
    def mean_absolute_percentage_error(y_true, y_pred):
        "Computes the mean absoulte percentage error" 
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    #compute evaluation metrics
    MSE = metrics.mean_squared_error(y_true, y_pred)
    MAE = metrics.mean_absolute_error(y_true, y_pred)
    RMSE = np.sqrt(metrics.mean_squared_error(y_true, y_pred))
    MAPE = mean_absolute_percentage_error(y_true, y_pred)
    R2 = metrics.r2_score(y_true, y_pred)

    eval_metrics = {'MSE':MSE, 'MAE':MAE,'RMSE':RMSE,'MAPE':MAPE,'R2':R2}

    if print_results is True:
        #print evaluation metrics
        print(f'MSE is : {MSE}')
        print(f'MAE is : {MAE}')
        print(f'RMSE is : {RMSE}')
        print(f'MAPE is : {MAPE}')
        print(f'R2 is : {R2}',end='\n\n')

    return eval_metrics

def featureCombinations(cols, mustHave=[], lag_features = None):
    """
    We perform feature combinations here. This takes 
    """
    if lag_features:
        replacing_dict = {'lags':lag_features,
                      'Q':['Q_1','Q_2','Q_3','Q_4'],
                      'M':['M_1','M_2','M_3','M_4','M_5','M_6','M_7','M_8','M_9','M_10','M_11','M_12']
                        }

  #create empty array for all combinations
    featureSets = []

  #include 
    for L in range(len(mustHave),len(cols)+1):
        for subset in itertools.combinations(cols,L):
            if all(x in subset for x in mustHave):
        #check if subset is empty
                if not subset:
          #featureSets.append(None)
                    pass
        
                elif lag_features:
                #this will run by consolodating lags and datetime features
                    replaced_features = [a for b in subset for a in replacing_dict.get(b,[b])]
                    featureSets.append(replaced_features)
                else:
                    featureSets.append(list(subset))
    return featureSets

def dictTodf(models_dict,df_cols = ['features','MAPE','R2']):
    """This function takes in the dictionary hosting all the models and then converts relevant info to a dataframe which is easily readable"""
    df_models = pd.DataFrame(models_dict).T[df_cols]
    return df_models

def plotResults(y_train, y_test, y_prediction):

    #convert y_prediction to a series
    y_prediction = pd.Series(y_prediction,index=y_test.index)
    
    fig, axes=plt.subplots(figsize=(9, 4))
    y_train.plot(ax=axes, label='train')
    y_test.plot(ax=axes, label='test')
    y_prediction.plot(ax=axes, label='predictions')
    axes.legend();
    fig.show()
    return fig

def mergeResults(output_path, df_lstm=None, df_lr=None, df_prophet=None):
    df_final = pd.concat([df_lstm,df_lr,df_prophet])
    df_final.sort_values(by='MAPE',ascending=True,inplace=True)
    #display(df_final)
    #save df_final
    path = os.path.join(output_path,'results.csv')
    df_final.to_csv(path)

#Framework initializer class
class FrameworkOutput():
    "A class that organizes each run. When this runs, it will create a new folder for each run"
    def __init__(self, input_file, output_path, model_types = ['Linear_Regression','Prophet','LSTM']):
        self.input_file = input_file
        self.output_path = output_path
        self.model_types = model_types
        print('Making directories')
        
        try:
            os.mkdir(self.output_path)
        except OSError as error: 
            print(error) 
        
        #create folders for each model
        for model in self.model_types:
            path = os.path.join(self.output_path, str(model))
            os.mkdir(path)

#Dataloading class
#class for loading the data
class DataLoader():
    """A custom class that will preprocess the data so that it can
    easily be converted for separate models"""

    def __init__(self,output_path,filePath,date_col,target_col,numerical_cols, categorical_cols=None, remove_cols = None,
                model_types = ['Linear_Regression','Prophet','LSTM'], 
                scaler_type=None,scale_target=False,lag_steps=None,datetime_features=['Q','M'],
                generate_dummy_cols = True, holiday_country=None, test_split=0.2,
                train_window_size_lstm = 12, forecast_steps_lstm = 1, 
                ):
        self.output_path = output_path
        self.df = pd.read_csv(filePath)
        self.date_col = date_col
        self.target_col = target_col
        self.model_types = model_types
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        self.remove_cols = remove_cols
        self.datetime_features = datetime_features
        self.generate_dummy_cols = generate_dummy_cols
        self.holiday_country = holiday_country
        self.test_split = test_split 
        #final dataset that is common to all models
        self.df_final = self.preProcess()
        self.n_samples = len(self.df_final)
        self.original_features = list(self.df_final.columns.values)
        self.original_features.remove('ds')
        self.original_features.remove('y')
        self.df_LR = None #setting this to none to see whether it has changed or not for LSTMs

        if scaler_type:
            self.scaler_type = scaler_type
            self.scalers = {'minmax':MinMaxScaler(),'standard':StandardScaler()}
            self.df_scaled, self.features_scaler = self.scaleFeatures()
                #scale the target if required
        if scale_target:
            self.scale_target = scale_target
            self.target_scaler = self.scaleTarget()
        

        #let us generate the Linear regression and Prophet dataframes
        for model in self.model_types:
            print(f'Preparing data for {model}')
            if model == 'Linear_Regression':
                #separate LR dataframe from general df
                self.df_LR = self.df_scaled.copy()
                
                if lag_steps:
                    self.lag_steps = lag_steps
                    self.lag_features = self.addLagSteps()
                #extract date time features
                if datetime_features:
                    self.df_LR = self.extractDateTimeFeatures(self.df_LR)


                #create final dataframe
                self.X_train, self.y_train, self.X_test, self.y_test = self.splitData(model)
                print(output_path)
                
                df_LR_path = os.path.join(output_path,'Linear_Regression','linearRegression_data.csv')
                self.df_LR.to_csv(df_LR_path)
            
            elif model == 'Prophet':
                self.df_train_FB, self.df_test_FB = self.splitData(model)

            
            elif model == 'LSTM':
                self.df_LSTM = self.df_scaled.copy()
                #now lets add datetime features
                self.df_LSTM  = self.extractDateTimeFeatures(self.df_LSTM)
                df_LSTM_path = os.path.join(output_path,'LSTM','lstm_data.csv')
                self.df_LSTM.to_csv(df_LSTM_path)
                
                #self.train_window_size_lstm = train_window_size_lstm
                #self.forecast_steps_lstm = forecast_steps_lstm
                #initiate dataframe to be used
                #self.X_lstm, self.y_lstm = self.prepareLSTMData()
        
        #writing files
        #write the main dataframe
        df_final_path = os.path.join(output_path,'processed_data.csv')
        self.df_final.to_csv(df_final_path)
                
    def preProcess(self):
        """
        This function makes simple changes to the dataframe
        1. Rename the columns to ds and y
        2. Convert date column type to datetime
        """
        df_final = self.df.copy()
        #rename columns
        df_final.rename(columns={self.date_col:'ds',self.target_col:'y'},inplace=True)
        #convert date column to datetime
        df_final['ds'] = pd.to_datetime(df_final['ds'])
        #sort the dataframe
        df_final.sort_values(by='ds')

        if self.remove_cols:
            df_final.drop(columns=self.remove_cols,inplace=True)

        return df_final
    
    def scaleFeatures(self):
        """Function that scales the data"""
        print('Scaling data')
        df = self.df_final.copy()
        
        #initiate scaler
        scaler = self.scalers[self.scaler_type]

        if self.numerical_cols:
            #only do this operation if numerical columns present
            scaler = scaler.fit(df[self.numerical_cols])
            df[self.numerical_cols] = scaler.transform(df[self.numerical_cols])
        
        print('Numerical columns scaled')

        return df, scaler

    def scaleTarget(self):
        """A function to scale the target column"""
        print('Scaling the target')

        scaler = self.scalers[self.scaler_type]

        if self.scale_target is True:
            scaler = scaler.fit(self.df_scaled[['y']])
            self.df_scaled[['y']] = scaler.transform(self.df_scaled[['y']])

        return scaler
    
    def generateDummies(self,df, feature):
        """Takes in a categorical columns and then converts it to dummy columns"""
        dummies = pd.get_dummies(df[feature], prefix = feature)
        df = pd.concat([df,dummies],axis=1).drop(columns=[feature])

        return df
        
    def extractDateTimeFeatures(self,df):
        """Here we extract all datetime features. This will only work for LSTMs and Regression models"""
       
          #extracting date specific features
        for feature in self.datetime_features:
           
            if feature == 'Y':
                df[feature] = df['ds'].dt.year
            elif feature == 'Q':
                df[feature] = df['ds'].dt.quarter
            elif feature == 'M':
                df[feature] = df['ds'].dt.month
            elif feature == 'W':
               df[feature] = df['ds'].dt.week
            elif feature == 'DoW':
               df[feature] = df['ds'].dt.dayofweek
            elif feature == 'D':
                df[feature] =df['ds'].dt.day
            elif feature == 'H':
                df[feature] = df['ds'].dt.hour
        
            
            if self.generate_dummy_cols:
                df = self.generateDummies(df=df, feature=feature)
        
        return df
             
    def addLagSteps(self):
        """This function allows the user to add lag steps"""

        #fetch scaled y values
        y = pd.DataFrame(self.df_LR[['y']].values)
        #compute lag values by shifting incremently for the total range
        lags = pd.concat([y.shift(i) for i in range(1,self.lag_steps,1)],axis=1)
        #name the lags accordingly
        lags.columns = ['t-'+str(i) for i in range(1,self.lag_steps,1)]
        lag_features = list(lags.columns.values)
        #merge lag steps with dataset
        #we create a copy of the scaled version of the dataset
        
        self.df_LR = pd.concat([self.df_LR,lags],axis=1)
        #remove NA values which will be the first n rows (n = lag_steps)
        self.df_LR.dropna(inplace=True)

        return lag_features
    
    def generateHolidays(self):
        """This function impacts both, Linear Regression and Prophet as both can use holidays as a feature"""
        #computing holiday features
        if self.holiday_country:
            #looking at every date and checking whether it's a holiday or not
            self.df_final['isHoliday'] = pd.Series(self.df_final['ds']).apply(
                lambda x: holidays.CountryHoliday(self.holiday_country).get(x)).values.astype(bool).astype(int)
    
    def splitData(self,model_type):
        """Splits data based on the type of model that is passed to it"""
        if self.test_split < 1:
            #i.e if test split is given as a fraction instead of number of samples
            test_samples = round(self.n_samples*self.test_split)
            train_samples = self.n_samples - test_samples
        
        if model_type == 'Linear_Regression':
            #alter the the LR regression dataset
            print('Splitting data for Linear regression')
            #let us first set ds as index
            self.df_LR.set_index('ds',inplace=True)
            #seperate train and test
            train = self.df_LR.head(train_samples)
            test = self.df_LR.tail(test_samples)
            #seperate X and y
            X_train, X_test = train.drop(columns=['y']), test.drop(columns=['y'])
            y_train, y_test = train['y'], test['y']
            #let us print for sanity check
            print('Training samples: ',X_train.shape[0])
            print('Testing samples: ',X_test.shape[0])
            print('Number of features: ', X_train.shape[1])

            return X_train, y_train, X_test, y_test
        
        elif model_type == 'Prophet':
            print('Splitting data for Prophet')
            df_train, df_test = self.df_final.head(train_samples), self.df_final.tail(test_samples)
            return df_train, df_test
        
        else:
            print('Performing split for LSTMs')
    
#LSTM class
class modelLSTM():
    """A class for modelling LSTMs. Here, we do more than modeling; we also prepare different datasets due to the nature of LSTMs"""
    def __init__(self,output_path,df_LSTM, max_neurons, min_neurons, max_layers, forecast_timesteps, past_timesteps, train_test_split=0.8):
        self.output_path = os.path.join(output_path,'LSTM')
        self.df = df_LSTM
        self.max_neurons = max_neurons
        self.min_neurons = min_neurons
        self.max_layers = max_layers
        self.forecast_timesteps = forecast_timesteps
        self.past_timesteps = past_timesteps
        self.train_test_split = train_test_split

        #dataset combinations
        self.dataset_combinations = self.prepareDataCombinations()

        #architecture combinations
        self.architecture_combinations = self.makeArchCombinations()

        #here we iterate over all combinations and get a 
        self.summary, self.top_model, self.top_model_ID, self.top_model_history = self.iterateLstm()

        #let us now save the dataframe
        summary_path = os.path.join(self.output_path,'Lstm_results.csv')
        self.summary_df = dictTodf(self.summary,df_cols = ['Features','MAPE','Forecast_type','Training_window','Forecast_timesteps','Neurons'])
        self.summary_df.sort_values('MAPE',inplace=True, ascending=True)
        summary_path = os.path.join(self.output_path,'Lstm_results.csv')
        self.summary_df.to_csv(summary_path)
    
    def prepareDataCombinations(self,):
        import itertools
        dataset_combinations = list(itertools.product(*[self.past_timesteps,self.forecast_timesteps,['univariate','multivariate'],['discrete','non_discrete']]))
        return dataset_combinations
            
    def makeArchCombinations(self):

        architectures = []
        #append 1 and 2 layer architectures
        architectures.append([self.max_neurons])
        if self.max_layers == 2:
            architectures.append([self.max_neurons,self.min_neurons])
        if self.max_layers >= 3:

            for i in range(2,self.max_layers):
                arch = []
                interval = int((self.max_neurons - self.min_neurons)/(i))
                arch = list(range(self.max_neurons, self.min_neurons-1, -interval))
                architectures.append(arch)
        
        return architectures
    
    def prepareData(self, dataset_config):
        "Generates X and y for LSTM model"
        data = self.df.copy()
        data = data.set_index('ds')

        #set parameters
        
        #set parameter for past timesteps
        train_window_size, forecast_steps = dataset_config[0], dataset_config[1]
        features, discrete = dataset_config[2], dataset_config[3]

        X, y = [],[]
        window_start = 0
        if discrete == 'discrete':
            move_window = forecast_steps
        else:
            move_window = 1
        
        if features == 'univariate':
            data = data[['y']]
        
        samples = round((len(data) - train_window_size - forecast_steps) / move_window)

        for i in range(samples):
            train_window_end = window_start +train_window_size
            forecast_window_end = train_window_end + forecast_steps

            if forecast_window_end > len(data):
                break
            
            train = data[window_start:train_window_end].values
            forecast = data[train_window_end:forecast_window_end]['y'].values

            X.append(train)
            y.append(forecast)

            #increment window start
            window_start+=move_window

        return np.asarray(X), np.asarray(y)
    
    def fitLstm(self,X,y,train_samples, epochs, n_neurons,n_batch,n_layers):

        train_X, test_X = X[:train_samples],X[train_samples:]
        train_Y, test_Y = y[:train_samples],y[train_samples:]
    
    # initiate model
        model = Sequential()
        if n_layers == 1:
            model.add(LSTM(n_neurons[0], input_shape=(train_X.shape[1], train_X.shape[2])))
        
        else:
            
            print('Number of layers: ',n_layers)
            print('Arch: ',n_neurons)

            model.add(LSTM(n_neurons[0], input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True))
            for i in range(1,n_layers-1):
                model.add(LSTM(n_neurons[i], return_sequences=True))
            
            model.add(LSTM(n_neurons[-1]))
            
        model.add(Dense(train_Y.shape[1]))
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['acc'])
        history = model.fit(train_X,train_Y,epochs=30,verbose=0, validation_data = (test_X, test_Y),shuffle=False)

        return model, history, test_X, test_Y
    
    def forecastLstm(self, model, test_X, test_Y):

        forecast = model.predict(test_X)
        #perhaps write logic for consolodating 
        
        return forecast
    
    def evaluateLstm(self, forecast, test_Y):
        mape = keras.losses.mean_absolute_percentage_error(test_Y, forecast).numpy().mean()
        return mape
    
    def iterateLstm(self):
        "Here we iterate over all possible dataset combinations and architectures"
        summary = dict()
        total_models = len(self.dataset_combinations)*len(self.architecture_combinations)
        print(f'Training {total_models} models')

        #set lowest mape to 100
        lowest_mape = 100

        #initializing top model
        top_model, top_model_ID = None, None

        counter = 1

        for dataset_config in self.dataset_combinations:
            
            #get dataset information
            training_window, forecast_timesteps = dataset_config[0], dataset_config[1]
            features = dataset_config[2]
            forecast_type = dataset_config[3]

            #generate dataset according to configuration
            X,y = self.prepareData(dataset_config)
            
            for architecture in self.architecture_combinations:
                n_layers = len(architecture)
                train_samples = round(self.train_test_split*len(X))
                model, history, test_X, test_Y = self.fitLstm(X,y,train_samples,10,architecture,1,n_layers)
                forecast = self.forecastLstm(model ,test_X, test_Y)
                mape = self.evaluateLstm(test_Y,forecast)

                #let us pass this in the dictionary
                model_ID = 'LSTM_' + str(counter)
                summary[model_ID] = {'Training_window': training_window,
                                     'Forecast_timesteps': forecast_timesteps,
                                     'Features':features,
                                     'Forecast_type':forecast_type,
                                     'LSTM_layers':n_layers,
                                     'Neurons':architecture,
                                     'MAPE':mape}

                if mape < lowest_mape:
                    #assign new global lowest mape
                    lowest_mape = mape
                    top_model = model
                    top_model_ID = model_ID
                    top_model_history = history
                
                print(f'{counter} of {total_models} trained. So far, the best model is {top_model_ID} with a mape of {lowest_mape}.')
                counter+=1
                #print(model.summary())
        
        return summary, top_model, top_model_ID, top_model_history

class modelProphet():
    """A class for Prophet"""
    def __init__(self, output_path, df_train, df_test):
        self.output_path = os.path.join(output_path,'Prophet')
        self.df_train = df_train
        self.df_test = df_test
        self.features = list(self.df_train.columns.values)
        self.y_true = list(self.df_test['y'].values)
        self.feature_sets = featureCombinations(self.features,mustHave=['ds','y'])

        #self.prophetModel = self.fitProphet(self.df_train,self.df_test)
        #self.predictions = self.forecastProphet(self.prophetModel,self.df_train,self.df_test)
        #self.mets = self.evaluateProphet(self.predictions)

        self.summary, self.top_model, self.top_model_id = self.iterateProphet()

        self.summary_df = dictTodf(self.summary)
        self.summary_df.sort_values('MAPE',inplace=True, ascending=True)
        summary_path = os.path.join(self.output_path,'Prophet_results.csv')
        self.summary_df.to_csv(summary_path)

        #plot results
        os.mkdir(os.path.join(self.output_path,'plots'))
        #prediction plots
        prediction_plot_path = os.path.join(self.output_path,'plots','prediction_plot.png')
        train = self.df_train.set_index('ds')['y']
        test = self.df_test.set_index('ds')['y']
        predictions = self.summary[self.top_model_id]['predictions'].values
        prediction_plot = plotResults(train,test, predictions)
        prediction_plot.savefig(prediction_plot_path)

        #plotting regressor coefficients
        feature_imp_plot_path = os.path.join(self.output_path,'plots','feature_imp.png')
        featureImp = self.summary[self.top_model_id]['featureImp'].set_index('regressor')
        plot = featureImp.plot(kind='bar')
        fig = plot.get_figure()
        fig.savefig(feature_imp_plot_path,bbox_inches = 'tight')

    def fitProphet(self, train, test,univariate=False):
        
        #initiate model
        model = Prophet()
        #check for uni-variate case
        if univariate:
            #univariate case
            regressor_coef = None
            model.fit(train)
        else:
            #multivariate case so add regressors
            regressors = list(train.columns.values)
            regressors = [regressor for regressor in regressors if regressor not in ['ds','y']]
            for regressor in regressors:
                model.add_regressor(regressor)
            model.fit(train)
            #get the regressor coefficients
            regressor_coef = regressor_coefficients(model)
            regressor_coef = regressor_coef[['regressor', 'coef']].sort_values('coef',ascending=False)
        return model, regressor_coef
    
    def forecastProphet(self, model, test,univariate):
        """Here we forecast on the test set for evaluation"""
        #let us forecast on test first 
        if univariate is True:
            
            #select only the date column
            test_dates = test['ds'].reset_index()
        else:
            #selct date and regressor columns by simply dropping y
            test_dates = test.drop(columns=['y'])

        forecast_test = model.predict(test_dates)

        y_predictions = forecast_test['yhat']

        return y_predictions
    
    def evaluateProphet(self,y_predictions):
        
        eval_metrics = evaluation_metrics(self.y_true,y_predictions)
        return eval_metrics
    
    def iterateProphet(self):
        """Here we go over all Prophet models"""
        summary = dict()
        counter = 1
        lowest_mape = 1000
        for feature_set in self.feature_sets:
            #select relevant features
            train, test = self.df_train[feature_set], self.df_test[feature_set]
            #do univariate check
            if len(train.columns) == 2:
                univariate = True
            elif len(train.columns) > 2:
                univariate = False
            else:
                print('Must provide at least a date and target column')
                break
            #fit, predict and evaluate
            model, regressor_coef = self.fitProphet(train, test, univariate)
           
            y_predictions = self.forecastProphet(model,test,univariate)
            eval_metrics = self.evaluateProphet(y_predictions)

            #populate summary dictionary
            model_ID = 'Prophet_' + str(counter)
            model_features = list(train.columns.values)
            #remove date and target column from feature names
            model_features.remove('ds')
            model_features.remove('y')
            summary[model_ID] = {'features':model_features,'predictions':y_predictions,'featureImp':regressor_coef,
                             'MSE':eval_metrics['MSE'],'MAE':eval_metrics['MAE'],'RMSE':eval_metrics['RMSE'],'MAPE':eval_metrics['MAPE'],'R2':eval_metrics['R2']}
            
            if eval_metrics['MAPE'] < lowest_mape:
                #set lowest mape to the new best model
                lowest_mape = eval_metrics['MAPE']
                #save this model
                top_model_id = model_ID
                top_model = model
        
            counter+=1
            
        return summary, top_model, top_model_id

class modelLR():
    """A class to model linear-regression"""
    def __init__(self,output_path, df_LR, X_train, y_train, X_test,y_test, original_features , lag_features = None, datetime_features = ['Q','M']):
        self.output_path = os.path.join(output_path,'Linear_Regression')
        self.df = df_LR
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.y_true = list(y_test.values)
        self.features = original_features + ['lags'] + datetime_features
        #compute feature combinations
        print('Doing feature combinations')
        self.feature_sets = featureCombinations(self.features,mustHave=[],lag_features=lag_features)
        #generate summary
        self.summary, self.top_model, self.top_model_id = self.iterateLR()
        #convert summary dictionary to dataframe
        self.summary_df = dictTodf(self.summary)
        self.summary_df.sort_values('MAPE',inplace=True, ascending=True)
        summary_path = os.path.join(self.output_path,'LR_results.csv')
        self.summary_df.to_csv(summary_path)

        #plot results
        os.mkdir(os.path.join(self.output_path,'plots'))
        #prediction plots
        prediction_plot_path = os.path.join(self.output_path,'plots','prediction_plot.png')
        prediction_plot = plotResults(self.y_train,self.y_test,self.summary[self.top_model_id]['predictions'])
        prediction_plot.savefig(prediction_plot_path)

        #plotting feature importance
        feature_imp_plot_path = os.path.join(self.output_path,'plots','feature_imp_plot.png')
        feature_imp_values = (self.summary[self.top_model_id]['featureImp'])
        feature_imp_index = (self.summary[self.top_model_id]['features'])
        feature_imp = pd.Series(data=feature_imp_values,index=feature_imp_index)
        feature_imp.sort_values(inplace=True,ascending=False)
        feature_imp_fig = feature_imp.plot(kind='bar',title='Feature Importance',xlabel='Features',legend=False).get_figure()
        feature_imp_fig.savefig(feature_imp_plot_path,bbox_inches = 'tight')
   
    def fitLR(self,X_train,y_train):
        model = LinearRegression()
        model.fit(X_train,y_train)
        feature_importance = model.coef_
        return model, feature_importance
    
    def forecastLR(self, model, X_test):
        y_predictions = model.predict(X_test)
        return y_predictions
    
    def evaluateLR(self, y_predictions):
        eval_metrics = evaluation_metrics(self.y_true,y_predictions)
        return eval_metrics
    
    def iterateLR(self):
        summary = dict()
        counter = 1
        lowest_mape = 1000

        for featureSet in self.feature_sets:
            
            #select only relevant columns
            X_train, X_test = self.X_train[featureSet],self.X_test[featureSet]
            #fit model
            model, feature_importance = self.fitLR(X_train,self.y_train)
            #make predictions
            y_predictions = self.forecastLR(model,X_test)
            #evaluate model
            eval_metrics = self.evaluateLR(y_predictions)


            
            #populate summary dictionary
            model_ID = 'Linear_' + str(counter)
            model_features = list(X_train.columns.values)
            
            summary[model_ID] = {'features':model_features,'predictions':y_predictions,'featureImp':feature_importance,
                             'MSE':eval_metrics['MSE'],'MAE':eval_metrics['MAE'],'RMSE':eval_metrics['RMSE'],'MAPE':eval_metrics['MAPE'],'R2':eval_metrics['R2']}
        
            #lets save the best model and plot it too
            if eval_metrics['MAPE'] < lowest_mape:
                #set lowest mape to the new best model
                lowest_mape = eval_metrics['MAPE']
                #save this model
                top_model_id = model_ID
                top_model = model
            
            counter+=1
            # print('Model trained')
    
        return summary, top_model, top_model_id

