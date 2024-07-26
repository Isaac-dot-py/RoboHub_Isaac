import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI()


df = pd.read_csv('heartData.csv') 

averages = df.mean()

def create_prompt(averages):
    prompt = (f"Based on the following average values from a dataset of heart disease indicators, "
              f"provide insights on the general risk of heart disease:\n"
              f"Average Age: {averages['age']:.2f}\n"
              f"Average Sex (1 = male; 0 = female): {averages['sex (1 = male; 0 = female)']:.2f}\n"
              f"Average Chest Pain Type: {averages['chest_pain_type']:.2f}\n"
              f"Average Resting Blood Pressure: {averages['resting_blood_pressure']:.2f} mm Hg\n"
              f"Average Serum Cholestoral: {averages['serum_cholestoral']:.2f} mg/dl\n"
              f"Average Fasting Blood Sugar > 120 mg/dl: {averages['fasting_blood_sugar_&gt; 120 mg/dl (1 = true; 0 = false)']:.2f}\n"
              f"Average Resting ECG: {averages['restecg']:.2f}\n"
              f"Average Maximum Heart Rate Achieved: {averages['maximum heart rate achieved']:.2f}\n"
              f"Average Exercise Induced Angina: {averages['exercise induced angina (1 = yes; 0 = no)']:.2f}\n"
              f"Average Oldpeak = ST Depression Induced by Exercise Relative to Rest: {averages['oldpeak = ST depression induced by exercise relative to rest']:.2f}\n"
              f"Average Slope of the Peak Exercise ST Segment: {averages['the slope of the peak exercise ST segment']:.2f}\n"
              f"Average Number of Major Vessels Colored by Fluoroscopy: {averages['number of major vessels (0-3) colored by flourosopy']:.2f}\n"
              f"Average Thal: {'Normal' if averages['thal: 0 = normal; 1 = fixed defect; 2 = reversable defect'] == 0 else 'Defect'}\n"
              f"Provide insights on the general risk of heart disease based on these averages.")

    return prompt

prompt = create_prompt(averages)

def get_insights(prompt):
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # You can use any model suited for summarization
        messages=[
            {"role": "system", "content": "You are the master at inferring data based off of means"},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1000,
        temperature=0.7
    )
    
    return response.choices[0].message.content

insights = get_insights(prompt)

print("Insights Based on Averages:")
print(insights)