from flask import Flask, request, jsonify
import pandas as pd
import HazardRecommModel
import pickle
import numpy as np
import base64

app = Flask(__name__)

# Load the embeddings dictionary from the binary file using pickle
with open('embeddings_df.pkl', 'rb') as f:
    df = pickle.load(f)

location_data = pd.read_csv('Location_work_permit.csv')

locations = df['New_Location'].unique().tolist()
wp = location_data['Work_permit'].unique().tolist()

@app.route('/recommendations', methods=['POST'])
def recommendation():
    data = request.get_json()
    location = data.get('location')
    WP = data.get('work_permit')
    activities = data.get('activities')
    chemicals = data.get('chemicals')

    combined_input = ",".join(filter(lambda x: x != "", [activities, chemicals, WP]))
    activities = combined_input.split(',')

    locations_expanded = [location] * len(activities)

    df_input = pd.DataFrame({'Location': locations_expanded,
                            'Activity': activities})

    recommendation = HazardRecommModel.get_recommendation(df_input, location)
    # print(recommendation.columns)
    result = []
    unique_hazards = recommendation['Hazards'].unique()[:10]
    for hazard in unique_hazards:
        hazard_category = recommendation.loc[recommendation['Hazards'] == hazard, 'Hazard Category'].iloc[0]
        risk_category = recommendation.loc[recommendation['Hazards'] == hazard, 'Risk_Category'].iloc[0]
        observations = recommendation[recommendation['Hazards'] == hazard][['Observation Details']]
        observations_html = observations.to_html(index=False, border=0, classes=["table", "table-hover"])
        Recommendations = recommendation.loc[recommendation['Hazards'] == hazard, 'Recommendations'].iloc[0]
        # print(recommendations_html)
        result.append({
            "Hazard Category": hazard_category,
            "Hazards": hazard,
            "Observation Details": observations_html,
            "Risk Category": risk_category,
            "Recommendations": Recommendations
        })

    return jsonify({"recommendations": result})

@app.route('/data')
def get_data():
    data = pd.read_pickle('embeddings_df.pkl')
    # Convert ndarray to list
    data = data.applymap(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
    return jsonify(data.to_dict(orient='records'))

@app.route('/download', methods=['POST'])
def download_recommendations():
    data = request.get_json()
    recommendation_df = pd.DataFrame(data["recommendations"])
    csv = recommendation_df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="recommendation.csv">Download CSV</a>'
    return href

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5174)