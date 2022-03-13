import sys
import pandas as pd
from sqlalchemy import create_engine
import sqlite3


def load_data(messages_filepath, categories_filepath):
    '''
    Loads the dataset using filepath and Perform Preprocessing steps
    
    INPUT:
    messages_filepath: message dataset file path
    categories_filepath: categories dataset file path
    
    OUTPUT:
    df: Preprocessed Dataframe which is ready for Further Steps
    '''
    # read in file
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    #messages.head() 
    
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    #categories.head()
    
    #merge datasets
    merged_df = pd.merge(messages,categories,how='inner',on='id')
    
    #Spliting the Categories column and storing the individual columns in a separate df
    individual_categories = merged_df['categories'].str.split(';',expand=True)
    
    #Selecting the First Column of df_categories df
    row=individual_categories[:1].values[0]
    
    #Appending the New Column names in a separate List
    categories_column=[]
    for val in row:
        categories_column.append(val.split('-')[0])
        
    #Renaming the Categories Column in ind_categories dataframe
    individual_categories.columns = categories_column
    
    #Create a dataframe of the 36 individual category columns
    #Concating the original df with individual_categories df
    
    df = pd.concat([merged_df,individual_categories],axis=1)
    df.drop(['categories'],axis=1,inplace=True)
    
    return df
     

def clean_data(df):
    '''
    this function does the following:
    - split the attribute values and only store the numeric part i.e 1 or 0
    - Remove the duplicates
    - Check whether any column contains values other than 1 or 0
    '''
    target_names = ['related', 'request', 'offer', 'aid_related', 'medical_help','medical_products', 'search_and_rescue', 'security', 
                'military','child_alone', 'water', 'food', 'shelter', 'clothing', 'money','missing_people', 'refugees', 'death', 
                'other_aid','infrastructure_related', 'transport', 'buildings', 'electricity','tools', 'hospitals', 'shops', 
                'aid_centers','other_infrastructure','weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
                'other_weather', 'direct_report']
    
    #Keeping only 1 and 0 in the rows using loops and split
    for column in df:
        if column in target_names:
            # set each value to be the last character of the string
            df[column] = df[column].str.split('-',expand=True)[1]
            # convert column from string to numeric
            df[column] = df[column].astype(int)
    
    # drop duplicates
    df = df.drop_duplicates()
    
    #Replacing the values present in 'related' from '2' to '1'
    #df.related.unique()
    df.replace(2,1,inplace = True)
    
    return df

def save_data(df, database_filename):
    '''
    This Function stores the dataframe in a Database Table
    '''
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('DisasterResponseTable', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()