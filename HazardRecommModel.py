import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer,util
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from joblib import Parallel, delayed


# Make sure to download NLTK data
import nltk
nltk.download('wordnet')


#Similarity threshold
sim_thresh=0.25

#defining value of top recommendations
top_recomm_val = 20

#defining value to multiply into keyword_proportion(here we are doubling the value)
keyword_weight=2

# Load model once
model = SentenceTransformer('all-MiniLM-L6-v2')


# Load the embeddign dictionary from the binary file using pickle
with open('embeddings_df.pkl', 'rb') as f:
    loaded_data = pickle.load(f)

def CapitalizingLetter(df):
    """
    This function, CapitalizingLetter(), takes a pandas DataFrame as input and returns a modified DataFrame.

    The function capitalizes the first letter of the string in selected columns of the DataFrame.
    The input DataFrame is modified in place, and the modified DataFrame is returned.

    Args:
    df (pandas DataFrame): Input DataFrame to be modified

    Returns:
    pandas DataFrame: Modified DataFrame with selected columns capitalized
    
    """
    print("Inside CapitalizingLetter")
    df['Observation Details'] = df['Observation Details'].str.capitalize()
    df['HazardA'] = df['HazardA'].str.capitalize()
    df['HazardB'] = df['HazardB'].str.capitalize()
    df['HazardC'] = df['HazardC'].str.capitalize()
    df['Hazard Category'] = df['Hazard Category'].str.capitalize()
    df['Risk_Category'] = df['Risk_Category'].str.capitalize()
    # df['New_Location'] = df['New_Location'].str.capitalize()
    # df['Location Category'] = df['Location Category'].str.capitalize()
    df['Nature of Work'] = df['Nature of Work'].str.capitalize()
    df['Associated Hazard'] = df['Associated Hazard'].str.capitalize()
    return df    

# capitalizing first letter
loaded_data = CapitalizingLetter(loaded_data)

# Access the sentences and embeddings from the loaded dictionary
obs_embd = loaded_data['Obs_embd']

def preprocess(df):
    """
    Preprocesses the text data in the input DataFrame `df` by removing punctuation,
    converting all text to lower case, and replacing certain characters with spaces.

    Args:
    - df: A DataFrame containing text data to be preprocessed.

    Returns:
    - tempArr: A list of preprocessed strings.
    """
    print("Inside preprocess")
    #set up punctuations we want to be replaced
    REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\|)|(\()|(\))|(\[)|(\])|(\%)|(\$)|(\>)|(\<)|(\{)|(\})")
    REPLACE_WITH_SPACE = re.compile("(<br\s/><br\s/?)|(-)|(/)|(:).")
    tempArr = []
    for line in df:
        # remove puctuation
        tmpL = REPLACE_NO_SPACE.sub("", line.lower()) # convert all tweets to lower cases
        tmpL = REPLACE_WITH_SPACE.sub(" ", tmpL)
        tempArr.append(tmpL)
    return tempArr

# Embeddings 


# function to get embedding on observations & input
def get_embeddings_miniLM(text,model=model):
    """
    Encodes a given text using a pre-trained MiniLM model and returns the corresponding embeddings.

    Args:
        text (str): The input text to be encoded.
        model (MiniLM): A pre-trained MiniLM model from the transformers library.

    Returns:
        corpus_embeddings (numpy.ndarray): A 2D numpy array of shape (1, embedding_size) containing the embeddings of the input text.
        Each row represents an embedding vector of size 'embedding_size', where 'embedding_size' is the number of dimensions of the MiniLM
        model.

    Raises:
        None
    """
    print("Inside get_embeddings_miniLM")
    corpus_embeddings = model.encode(text)
    return corpus_embeddings

def get_location_df(Location):
    """
    Retrieves two dataframes from the loaded_data dataframe, one containing all rows with a 'New_Location'
    column value equal to the input Location parameter, and another containing all rows with a 'New_Location'
    column value not equal to the input Location parameter. The resulting dataframes are returned as a tuple.
    
    Parameters:
    -----------
    Location : str
        The value of the 'New_Location' column to use as the filter criterion for one of the resulting dataframes.
        
    Returns:
    --------
    A tuple of two pandas dataframes:
    - df_bldg : a dataframe containing all rows of loaded_data where the 'New_Location' column matches the input Location parameter.
    - df_others : a dataframe containing all rows of loaded_data where the 'New_Location' column does not match the input Location
    parameter.

    """

    print('Inside get_location_df')
    # two separate dataframes both including & excluding building 14 - !!!!!!!!!!!!Need to Update This code
    df_bldg = loaded_data[loaded_data['New_Location']==Location]
    df_others = loaded_data[loaded_data['New_Location']!=Location]
    return df_bldg,df_others

### Cosine Similarity on all-MiniLM-L6-v2 embedding

def cos_sim_miniLM(input_data, loaded_data, n_jobs=-1):
    """
    Calculates the cosine similarity between the input_data and loaded_data using parallel processing.
    
    Args:
    - input_data: A pandas DataFrame containing the input data.
    - loaded_data: A pandas DataFrame containing the pre-loaded data.
    - n_jobs: An integer indicating the number of CPU cores to be used for parallel processing. Default is -1, which
              means using all available cores.
    
    Returns:
    - A pandas DataFrame containing the input_data and loaded_data along with the cosine similarity between the
      'input_text_embedding' column of input_data and the 'Obs_embd' column of loaded_data. The DataFrame is sorted
      by 'Activity' column in descending order of cosine similarity, and duplicates are removed based on the
      'Activity' and 'Observation Details' columns.
    """
    print('MultiProcess-cos_sim_miniLM')

    # Preprocess input
    input_data['Activity'] = preprocess(input_data['Activity'])

    # Embedding on input text
    input_data['input_text_embedding'] = input_data['Activity'].apply(get_embeddings_miniLM)

    # Vectorize input text embeddings and loaded_data embeddings
    vec_col1 = np.vstack(input_data['input_text_embedding'])
    vec_col2 = np.vstack(loaded_data['Obs_embd'])

    # Compute cosine similarity between input & master observation using parallel processing
    cos_sim = Parallel(n_jobs=n_jobs)(delayed(cosine_similarity)(vec1.reshape(1, -1), vec_col2) for vec1 in vec_col1)
    
    
#     cos_sim = []
#     for i in range(len(vec_col1)):
#         vec1 = vec_col1[i]
#         vec2 = vec_col2[i]
#         similarity = cosine_similarity([vec1], [vec2])[0][0]
#         cos_sim.append(similarity)
    
    cos_sim = np.vstack(cos_sim)

    # Create a multi-index based on input_data and loaded_data
    multi_index = pd.MultiIndex.from_product([input_data.index, loaded_data.index], names=["input_idx", "loaded_idx"])

    # Create the final DataFrame with multi-index and cosine similarity values
    df_cos_sim = pd.DataFrame(cos_sim.flatten(), index=multi_index, columns=["cosine_similarity"]).reset_index()

    # Merge with input_data and loaded_data to get the final DataFrame
    df_merge = df_cos_sim.merge(input_data, left_on="input_idx", right_index=True).merge(loaded_data, left_on="loaded_idx", right_index=True)
    # print(df_merge.columns)
    # Select required columns and drop duplicates
    df_merge = df_merge[['Activity', 'Observation Details', 'HazardA', 'cosine_similarity','Recommendations','Risk_Cal','Risk_Category','Hazard Category']]
    df_merge = df_merge.rename(columns={'HazardA':'Hazards'})
    df_merge = df_merge.drop_duplicates(subset=['Activity', 'Observation Details'])

    # Sort the final DataFrame
    df_merge = df_merge.sort_values(by=['Activity', 'cosine_similarity'], ascending=False).reset_index(drop=True)

    return df_merge


### Cosine Similarity on all-MiniLM-L6-v2 embedding

# def cos_sim_miniLM1(input_data, loaded_data, n_jobs=-1):
#     print('MultiProcess-cos_sim_miniLM')

#     # Preprocess input
#     input_data['Activity'] = preprocess(input_data['Activity'])

#     # Embedding on input text
#     input_data['input_text_embedding'] = input_data['Activity'].apply(get_embeddings_miniLM)

#     # Vectorize input text embeddings and loaded_data embeddings
#     vec_col1 = np.vstack(input_data['input_text_embedding'])
#     vec_col2 = np.vstack(loaded_data['Obs_embd'])    

#     cos_sim = []
#     for i in range(len(vec_col1)):
#         vec1 = vec_col1[i]
#         vec2 = vec_col2[i]
#         similarity = cosine_similarity([vec1], [vec2])[0][0]
#         cos_sim.append(similarity)

#     cos_sim = np.vstack(cos_sim)
    
#     # Create a multi-index based on input_data and loaded_data
#     multi_index = pd.MultiIndex.from_product([input_data.index, loaded_data.index], names=["input_idx", "loaded_idx"])

#     # Create the final DataFrame with multi-index and cosine similarity values
#     df_cos_sim = pd.DataFrame(cos_sim, index=multi_index, columns=["cosine_similarity"]).reset_index()

#     # Merge with input_data and loaded_data to get the final DataFrame
#     df_merge = df_cos_sim.merge(input_data, left_on="input_idx", right_index=True).merge(loaded_data, left_on="loaded_idx", right_index=True)

#     # Select required columns and drop duplicates
#     df_merge = df_merge[['Activity', 'Observation Details', 'Hazards', 'cosine_similarity']]
#     df_merge = df_merge.drop_duplicates(subset=['Activity', 'Observation Details'])

#     # Sort the final DataFrame
#     df_merge = df_merge.sort_values(by=['Activity', 'cosine_similarity'], ascending=False).reset_index(drop=True)
    
#     return df_merge


# function to find out observations against each activity based on keyword search
def search_keywords(observations, keywords):
    """
    Search for a list of keywords in a list of observations.

    Args:
    - observations (list): A list of strings representing the observations to search through.
    - keywords (list): A list of strings representing the keywords to search for.
    
    Returns:
    - results (list): A list of strings representing the observations that contain any of the specified keywords. 
    
    The function first converts each keyword to its lemma form using WordNetLemmatizer. It then searches each observation for any of the
    lemmatized keywords. If a single-word keyword is found in the observation, the observation is added to the results list. If a multi-word
    keyword is found in the observation, the observation is also added to the results list. The function is case-insensitive and ignores
    punctuation.
    """
    print("Inside search_keywords")
    lemmatizer = WordNetLemmatizer()
    keyword_lemmas = {lemmatizer.lemmatize(keyword, wordnet.VERB) if ' ' not in keyword else keyword.lower() for keyword in keywords}
    results = []

    for obs in observations:
        obs_words = re.findall(r'\w+', obs.lower())
        obs_lemmas = {lemmatizer.lemmatize(word, wordnet.VERB) for word in obs_words}

        # Check single-word keywords
        if keyword_lemmas.intersection(obs_lemmas):
            results.append(obs)
        else:
            # Check multi-word keywords
            for keyword in keyword_lemmas:
                if ' ' in keyword and keyword in obs.lower():
                    results.append(obs)
                    break

    return results


#Getting similar obseravtion for Building and Non- Building Data
def sim_with_bldg(df_input,Location,sim_thresh):
    """
    This function takes input as a pandas dataframe, a string location and a similarity threshold. 
    The function returns the hazards that can occur based on the activities and observations made 
    in a particular location. The function calculates the similarity between input activities and 
    observations for building and for observations excluding building. The function returns 
    a dataframe that combines the observations based on cosine similarity and keyword search. 
    It then calculates the weight of keyword matches and embedding matches for each activity. 
    The function also applies weight to the keyword proportion and calculates the normalized 
    keyword and embedding proportions. The function then allocates the total hazards based on 
    activity-level proportions. 

    Parameters:
    df_input (pd.DataFrame): A pandas dataframe containing the input activities
    Location (str): A string representing the location of interest
    sim_thresh (float): A floating-point number representing the similarity threshold

    Returns:
    pd.DataFrame: A pandas dataframe that contains the observations based on the activities, along with obs count for each activity.
    """

    print('Inside sim_with_bldg')
    df_bldg,df_others = get_location_df(Location)
    #getting similarities b/w input activities & observations for buildings
    recomm_bld14 = cos_sim_miniLM(df_input,df_bldg)
    #getting similarities b/w input activities & observations excluding building 14
    recomm_others = cos_sim_miniLM(df_input,df_others)
    # adding bldg(yes/no) column
    recomm_bld14['bldg'] = 'yes'
    recomm_others['bldg'] = 'no'

    # combine bldg & non bldg observations 
    combined_obs = pd.concat([recomm_bld14, recomm_others])
    
    # filtering observations basis cosine similarity
    combined_obs = combined_obs[combined_obs['cosine_similarity']>=sim_thresh]
    
    # getting observations & activities list
    obs_lst = combined_obs['Observation Details'].unique().tolist()
    keyword_lst = combined_obs['Activity'].unique().tolist()

    # getting matched observations based on keyword search
    results = search_keywords(obs_lst, keyword_lst)
    match_obs = pd.DataFrame({'Observation Details': results})
    
    # adding keyword_based column & flagging keyword based observations
    match_obs['keyword_based'] = 'yes'
    
    # combining embedding & keyword based observations together
    combined_obs = combined_obs.merge(match_obs,on='Observation Details', how='left')
    combined_obs['keyword_based'] = combined_obs['keyword_based'].fillna('no')
    
    #sorting basis keyword matched observations
    combined_obs = combined_obs.sort_values(by=['keyword_based'],ascending=False)
    # dropping off duplicates basis activity & observations
    combined_obs = combined_obs.drop_duplicates(subset=['Activity','Observation Details']).reset_index(drop=True)
    
    # creating 2 columns 'keywrd_weight' & 'emb_weight', showing keyword & embedding based obs count for each activity
    keywrd_weight =  combined_obs[combined_obs['keyword_based']=='yes'].groupby('Activity')['Observation Details'].nunique()
    emb_weight =  combined_obs[combined_obs['keyword_based']=='no'].groupby('Activity')['Observation Details'].nunique()

    combined_obs['keywrd_weight'] = combined_obs['Activity'].map(keywrd_weight)
    combined_obs['emb_weight'] = combined_obs['Activity'].map(emb_weight)
    
    combined_obs['keywrd_weight'] = combined_obs['keywrd_weight'].fillna(0)
    combined_obs['emb_weight'] = combined_obs['emb_weight'].fillna(0)
    
    act_weight_kwrd_emb = combined_obs[['Activity','keywrd_weight','emb_weight']]
    act_weight_kwrd_emb = act_weight_kwrd_emb.drop_duplicates().reset_index(drop=True)
    
    # Capping on embedding_weight
    for i in range(len(act_weight_kwrd_emb)):
        if((act_weight_kwrd_emb['keywrd_weight'].iloc[i]!=0) and (act_weight_kwrd_emb['emb_weight'].iloc[i]>act_weight_kwrd_emb['keywrd_weight'].iloc[i]*2)):
            act_weight_kwrd_emb['emb_weight'].iloc[i] = act_weight_kwrd_emb['keywrd_weight'].iloc[i]*2
    
    
    # Calculate the proportions of keyword matches and embedding matches
    act_weight_kwrd_emb['keyword_proportion'] = act_weight_kwrd_emb["keywrd_weight"] / (act_weight_kwrd_emb["keywrd_weight"] + act_weight_kwrd_emb["emb_weight"])
    act_weight_kwrd_emb['embedding_proportion'] = act_weight_kwrd_emb["emb_weight"] / (act_weight_kwrd_emb["keywrd_weight"] + act_weight_kwrd_emb["emb_weight"])

    # Apply the weight to the keyword proportion
    act_weight_kwrd_emb["weighted_keyword_proportion"] = act_weight_kwrd_emb['keyword_proportion']*keyword_weight
    act_weight_kwrd_emb["sum_weighted_proportions"] = act_weight_kwrd_emb['weighted_keyword_proportion'] + act_weight_kwrd_emb['embedding_proportion']

    # Calculate the normalized keyword and embedding proportions
    act_weight_kwrd_emb["normalized_keyword_proportion"] = act_weight_kwrd_emb["weighted_keyword_proportion"] / act_weight_kwrd_emb["sum_weighted_proportions"]
    act_weight_kwrd_emb["normalized_embedding_proportion"] = act_weight_kwrd_emb['embedding_proportion'] / act_weight_kwrd_emb["sum_weighted_proportions"]

    # Calculate the sum of all normalized keyword and embedding proportions
    act_weight_kwrd_emb['sum_all_normalized_keyword_proportions'] = act_weight_kwrd_emb["normalized_keyword_proportion"].sum()
    act_weight_kwrd_emb['sum_all_normalized_embedding_proportions'] = act_weight_kwrd_emb["normalized_embedding_proportion"].sum()
    act_weight_kwrd_emb['sum_all_normalized_proportions'] = act_weight_kwrd_emb['sum_all_normalized_keyword_proportions'] + act_weight_kwrd_emb['sum_all_normalized_embedding_proportions']
    
    # Allocate the total hazards based on the activity-level proportions
    act_weight_kwrd_emb["keyword_hazards"] = round(top_recomm_val * (act_weight_kwrd_emb["normalized_keyword_proportion"] / act_weight_kwrd_emb['sum_all_normalized_proportions']))
    act_weight_kwrd_emb["embedding_hazards"] = round(top_recomm_val * (act_weight_kwrd_emb["normalized_embedding_proportion"] / act_weight_kwrd_emb['sum_all_normalized_proportions']))

    act_weight_kwrd_emb["obs_count"] = act_weight_kwrd_emb["keyword_hazards"] + act_weight_kwrd_emb["embedding_hazards"]
    
    act_weight_kwrd_emb = act_weight_kwrd_emb[['Activity','keywrd_weight','emb_weight','obs_count']]
    act_weight_kwrd_emb = act_weight_kwrd_emb.drop_duplicates().reset_index(drop=True)
    
    combined_obs = combined_obs.drop(columns={'keywrd_weight','emb_weight'})
    combined_obs = combined_obs.merge(act_weight_kwrd_emb,on='Activity',how='inner')


    return combined_obs


    
# ## finding weights or total observations for an activity based on embeddings
# def embeddingBasedweights(combined_obs, sim_thresh):
#     print("Inside embeddingBasedweights")
    
#     #merging bldg  & non-bldg  observations and filtering above 25% similarity
#     combined_obs = combined_obs[combined_obs['cosine_similarity']>=sim_thresh].reset_index(drop=True)
#     combined_obs = combined_obs.drop_duplicates(subset=['Observation Details'])
    
#     # calculating weights or total observations for an activity
#     combined_obs['weightage'] = combined_obs.groupby('Activity')['Observation Details'].transform('nunique')
    
#     # calculating activity wise bldg  - yes & no count
#     bldg_yes_count = combined_obs[combined_obs['bldg']=='yes'].groupby('Activity')['Observation Details'].nunique()
#     bldg_no_count =  combined_obs[combined_obs['bldg']=='no'].groupby('Activity')['Observation Details'].nunique()

#     combined_obs['bldg_yes_count'] = combined_obs['Activity'].map(bldg_yes_count)
#     combined_obs['bldg_no_count'] = combined_obs['Activity'].map(bldg_no_count)
    
#     combined_obs = combined_obs.fillna(0)
    
#     return combined_obs


# ## finding weights or total observations for an activity based on embeddings
# def keywordBasedweights(combined_obs, sim_thresh):
#     print("Inside keywordBasedweights")
    
#     #merging bldg  & non-bldg  observations and filtering above 25% similarity
#     combined_obs = combined_obs[combined_obs['cosine_similarity']>=sim_thresh].reset_index(drop=True)
#     combined_obs = combined_obs.drop_duplicates(subset=['Observation Details'])
    
#     obs_lst = combined_obs['Observation Details'].unique().tolist()
#     keyword_lst = combined_obs['Activity'].unique().tolist()
    
#     # getting matched observations based on keyword search
#     results = search_keywords(obs_lst, keyword_lst)
#     match_obs = pd.DataFrame({'Observation Details': results})
    
#     combined_obs = combined_obs.merge(match_obs,on='Observation Details', how='inner')
    
#     # calculating weights or total observations for an activity
#     combined_obs['weightage'] = combined_obs.groupby('Activity')['Observation Details'].transform('nunique')
    
#     # calculating activity wise bldg - yes & no count
#     bldg_yes_count = combined_obs[combined_obs['bldg']=='yes'].groupby('Activity')['Observation Details'].nunique()
#     bldg_no_count =  combined_obs[combined_obs['bldg']=='no'].groupby('Activity')['Observation Details'].nunique()

#     combined_obs['bldg_yes_count'] = combined_obs['Activity'].map(bldg_yes_count)
#     combined_obs['bldg_no_count'] = combined_obs['Activity'].map(bldg_no_count)
    
#     combined_obs = combined_obs.fillna(0)
    
#     return combined_obs


# Calculating combined obs count & weights for each activity 

def obsCount_calculation_table(combined_obs_df,top_recomm_val): 
    """
    This function takes in a combined observations dataframe, and a top recommendation value as input. It calculates the activity-wise
    weightage, building yes and no count, and observations count. It also calculates the ratio between observations count and building yes
    count. If any observation count is zero or negative, it replaces that count with one, and redistributes the excess observations among
    the remaining activities. Finally, it merges the input dataframe with the calculated weightage, building counts, and observations
    counts, and sorts the merged dataframe by activity, keyword-based observations, and cosine similarity. It returns the merged dataframe
    as output.
    
    :param combined_obs_df: A pandas dataframe containing the combined observations.
    :type combined_obs_df: pandas.core.frame.DataFrame
    
    :param top_recomm_val: A value representing the top recommendation count.
    :type top_recomm_val: int
    
    :return: A pandas dataframe containing the combined observations, merged with calculated weightage, building counts, and observations
    counts, sorted by activity, keyword-based observations, and cosine similarity.
    :rtype: pandas.core.frame.DataFrame
    """
    print("Inside obsCount_calculation_table") 
    weight_input = combined_obs_df[['Activity','weightage', 'bldg_yes_count', 'bldg_no_count', 'keyword_based', 'obs_count']]
    weight_input = weight_input.drop_duplicates().reset_index(drop=True)
    
    # adding total observations, bldg yes & no count of keyword & embedding based observations
    weight_input['weightage'] =  weight_input.groupby('Activity')['weightage'].transform('sum')
    weight_input['bldg_yes_count'] =  weight_input.groupby('Activity')['bldg_yes_count'].transform('sum')
    weight_input['bldg_no_count'] =  weight_input.groupby('Activity')['bldg_no_count'].transform('sum')
    weight_input = weight_input.drop(columns={'keyword_based'})
    weight_input = weight_input.drop_duplicates().reset_index(drop=True)
    
    # calculating activity wise weight% 
#     total = weight_input['weightage'].sum()
#     weight_input['weight%'] = weight_input['weightage']/total

#     # calculating observations count
#     # if total observations is less that top recomm val, then multiply weight% with total observations to get obs count 
#     if(total<top_recomm_val):
#         weight_input['obs_count'] =  (weight_input['weight%']*total).round(0)
#     # Else multiply weight% with top recomm val to get obs count    
#     else:
#         weight_input['obs_count'] =  (weight_input['weight%']*top_recomm_val).round(0)
        
    # calculating ratio between observations count & bldg yes count
    weight_input['bldg%'] = weight_input['obs_count']/weight_input['bldg_yes_count']
    
    if((weight_input['obs_count'].min())<=0):
        
        # count of obs_count whereever <= zero
        num_zeros = (weight_input['obs_count'] <= 0).sum()

        # Replace those values by 1 in obs_count
        weight_input.loc[weight_input['obs_count'] <= 0, 'obs_count'] = 1
        
        # Find the index of the maximum value of obs_count
        max_val_index = weight_input['obs_count'].idxmax()

        #Replace the maximum obs_count value by the (diff of max obs_count value & num_zeros)
        diff = (weight_input['obs_count'].max()) - num_zeros
        weight_input.loc[max_val_index, 'obs_count'] = diff

    # dropping off repeated columns
    combined_obs_df = combined_obs_df.drop(columns={'weightage', 'bldg_yes_count', 'bldg_no_count', 'obs_count'})
    
    combined_obs_df = combined_obs_df.merge(weight_input, on='Activity', how='inner')
    combined_obs_df = combined_obs_df.sort_values(by=['Activity','keyword_based','cosine_similarity'], ascending=False)
    combined_obs_df = combined_obs_df.drop_duplicates(subset=['Activity','Observation Details']).reset_index(drop=True)
    
    return combined_obs_df

# activity wise top observations based on obs count of each activity giving priority to bldg yes & keyword search
def topRecomm_weightedObs_bldgYes(activityObs_df):
    """
    This function takes a pandas DataFrame of activity observations as input and returns a filtered DataFrame of recommended observations
    based on certain conditions.

    Args:
    activityObs_df (pandas.DataFrame): A DataFrame containing columns 'Activity', 'Observation Details', 'keyword_based',
    'cosine_similarity', 'bldg', 'obs_count', 'bldg_yes_count', and 'bldg_no_count'.

    Returns:
    pandas.DataFrame: A filtered DataFrame of recommended observations based on certain conditions, sorted by 'keyword_based' and
    'cosine_similarity', and grouped by 'Activity'.

    Notes:
    The function applies different filtering conditions on the input DataFrame depending on the counts of 'bldg_yes_count', 'bldg_no_count',
    'obs_count', and 'bldg%'. The resulting DataFrame contains all observations from 'bldg_yes' if 'bldg% <= 0.5' or a mix of observations
    from 'bldg_yes' and 'bldg_no' based on a specific ratio of observations.
    """

    print("Inside topRecomm_weightedObs_bldgYes")
    topRecomm_weightedObs = pd.DataFrame()
    df_grp = activityObs_df.groupby('Activity')
    for k, val in df_grp:
        # getting dataframe for each activity
        df = df_grp.get_group(k)
        #keyword & cosine similarity wise sorting 
        df = df.sort_values(by=['keyword_based','cosine_similarity'], ascending=False).reset_index(drop=True)
        
        # if bldg yes count = 0, take observations from bldg no 
        if(df['bldg_yes_count'][0]==0):
            df = (df[df['bldg']=='no']).head(int(df['obs_count'][0]))
        # if bldg no count = 0, take observations from bldg yes    
        elif(df['bldg_no_count'][0]==0):
            df = (df[df['bldg']=='yes']).head(int(df['obs_count'][0])) 
        # if obs_count/bldg yes ratio is <=0.5, take all observations from bldg yes     
        elif(df['bldg%'][0]<=0.5):
            df = (df[df['bldg']=='yes']).head(int(df['obs_count'][0]))        
        else:
            div_obs_cnt = round(df['obs_count'][0]/2)
            # if bldg yes count is less than obs_count, then the balance observations will be taken from bldg no 
            if(df['bldg_yes_count'][0]<div_obs_cnt): 
                add_obs_to_bldgNo = div_obs_cnt - df['bldg_yes_count'][0]
                df_yes = (df[df['bldg']=='yes']).head(int(df['bldg_yes_count'][0]))
                df_no = (df[df['bldg']=='no']).head(int(df['obs_count'][0] - div_obs_cnt)+int(add_obs_to_bldgNo))
                df = pd.concat([df_yes,df_no])
           
            # if bldg no count is less than obs_count, then the balance observations will be taken from bldg yes    
            elif(df['bldg_no_count'][0]<div_obs_cnt): 
                add_obs_to_bldgYes = div_obs_cnt - df['bldg_no_count'][0]
                df_yes = (df[df['bldg']=='yes']).head(int(df['obs_count'][0] - div_obs_cnt)+int(add_obs_to_bldgYes))
                df_no = (df[df['bldg']=='no']).head(int(df['bldg_no_count'][0]))
                df = pd.concat([df_yes,df_no])
                
            # else divide the no. of observations & take both from bldg yes & no    
            else:
                df_yes = (df[df['bldg']=='yes']).head(div_obs_cnt)
                df_no = (df[df['bldg']=='no']).head(int(df['obs_count'][0] - div_obs_cnt))
                df = pd.concat([df_yes,df_no])

        topRecomm_weightedObs = topRecomm_weightedObs.append(df)
        topRecomm_weightedObs = topRecomm_weightedObs.drop_duplicates(subset=['Activity','Observation Details'])
        topRecomm_weightedObs = topRecomm_weightedObs.reset_index(drop=True)

    return topRecomm_weightedObs


def get_recommendation(input_df,Location):
    
    """
    Given an input dataframe `input_df` and a `Location` string, this function returns a recommendation dataframe
    based on the observations related to the specified location. It groups the observations based on activity and
    weights them according to their similarity with the specified location. It then calculates the count of
    observations based on building presence and absence, for both embedding and keyword-based observations. The
    top 20 recommendations are obtained and returned as a dataframe. The function accepts an optional parameter
    `sim_thresh` (default value 0.25) which specifies the similarity threshold for weighting the observations.

    Parameters:
    -----------
    input_df : pandas.DataFrame
        Input dataframe containing observations
    Location : str
        String representing the location for which recommendations are required
    sim_thresh : float, optional
        Threshold value for similarity weighting (default value: 0.25)

    Returns:
    --------
    pandas.DataFrame
        Dataframe containing the top 20 recommendations based on the specified location

    """

    print("Inside get_recommendation")

    combined_obs=sim_with_bldg(input_df,Location,sim_thresh) #Inside get_location_df
    
    #grouping based on activity
    act_grp = combined_obs.groupby('Activity')
    combined_obs_weights = pd.DataFrame()

    for k,val in act_grp:
        # take single dataframe for an activity
        df = act_grp.get_group(k)
        
        # sorting basis high similarity
        df = df.sort_values(by=['cosine_similarity'],ascending=False)    

        if(df['emb_weight'].iloc[0]>0):
            # creating dataframe and taking observations as per new emb weights
            df_emb = (df[df['keyword_based']=='no']).head(int(df['emb_weight'].iloc[0]))
            df_remain =  df[df['keyword_based']=='yes']
            df = pd.concat([df_emb,df_remain])

        #appending all activity dataframes into one
        combined_obs_weights = combined_obs_weights.append(df).drop_duplicates().reset_index(drop=True)

    #combined_obs_weights = combined_obs.copy()

    # mapping bldg yes & no count both for keyword & embedding based obs
    bldg_yes_count_keywrd =  combined_obs_weights[(combined_obs_weights['keyword_based']=='yes') &
                             (combined_obs_weights['bldg']=='yes')].groupby('Activity')['Observation Details'].nunique()

    bldg_no_count_keywrd =  combined_obs_weights[(combined_obs_weights['keyword_based']=='yes') &
                                 (combined_obs_weights['bldg']=='no')].groupby('Activity')['Observation Details'].nunique()

    bldg_yes_count_emb =  combined_obs_weights[(combined_obs_weights['keyword_based']=='no') &
                                 (combined_obs_weights['bldg']=='yes')].groupby('Activity')['Observation Details'].nunique()

    bldg_no_count_emb =  combined_obs_weights[(combined_obs_weights['keyword_based']=='no') &
                                 (combined_obs_weights['bldg']=='no')].groupby('Activity')['Observation Details'].nunique()

    combined_obs_weights['bldg_yes_count_keywrd'] = combined_obs_weights['Activity'].map(bldg_yes_count_keywrd)
    combined_obs_weights['bldg_no_count_keywrd'] = combined_obs_weights['Activity'].map(bldg_no_count_keywrd)
    combined_obs_weights['bldg_yes_count_emb'] = combined_obs_weights['Activity'].map(bldg_yes_count_emb)
    combined_obs_weights['bldg_no_count_emb'] = combined_obs_weights['Activity'].map(bldg_no_count_emb)

    
    # grouping basis activity & keyword based observations    
    combined_obs_weights_new = pd.DataFrame()
    df_grp = combined_obs_weights.groupby(['Activity','keyword_based'])

    for k,val in df_grp:
        df = df_grp.get_group(k)
        # finding activity wise total obs count(weightage), bldg yes & no count both for embedding & keyword based observations
        df['bldg_yes_count'] = df[(df['bldg']=='yes')]['Observation Details'].nunique()
        df['bldg_no_count'] = df[(df['bldg']=='no')]['Observation Details'].nunique()
        df['weightage'] = df['Observation Details'].nunique()

        combined_obs_weights_new = combined_obs_weights_new.append(df).reset_index(drop=True)

    #combined_obs_weights_new = combined_obs_weights_new.drop(columns={'keywrd_weight','emb_weight'})  


    # getting obs count matrix for both embedding & keyword based activity-observations
    combined_obs_count_df = obsCount_calculation_table(combined_obs_weights_new,top_recomm_val)
    # print(combined_obs.columns)
    
    # finally getting top 20 recommendations
    topRecomm_combined = topRecomm_weightedObs_bldgYes(combined_obs_count_df)
    topRecomm_combined = topRecomm_combined.drop(index=0)

    keywrd_obs_count =  topRecomm_combined[
    (topRecomm_combined['keyword_based']=='yes')].groupby('Activity')['Observation Details'].nunique()

    emb_obs_count =  topRecomm_combined[
        (topRecomm_combined['keyword_based']=='no')].groupby('Activity')['Observation Details'].nunique()

    bldg_yes_obs_count =  topRecomm_combined[
        (topRecomm_combined['bldg']=='yes')].groupby('Activity')['Observation Details'].nunique()

    bldg_no_obs_count =  topRecomm_combined[
        (topRecomm_combined['bldg']=='no')].groupby('Activity')['Observation Details'].nunique()

    topRecomm_combined['keywrd_obs_count'] = topRecomm_combined['Activity'].map(keywrd_obs_count)
    topRecomm_combined['emb_obs_count'] = topRecomm_combined['Activity'].map(emb_obs_count)
    topRecomm_combined['bldg_yes_obs_count'] = topRecomm_combined['Activity'].map(bldg_yes_obs_count)
    topRecomm_combined['bldg_no_obs_count'] = topRecomm_combined['Activity'].map(bldg_no_obs_count)

    topRecomm_combined = topRecomm_combined.sort_values(by=['Risk_Cal'],ascending=False).reset_index(drop=True)

    df_bldg = loaded_data[loaded_data['New_Location']==Location]

    print("Giving Recommendation...........")

    return topRecomm_combined
















