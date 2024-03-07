from django.shortcuts import render, HttpResponse
from rest_framework.views import APIView
from django.http import JsonResponse
from dotenv import load_dotenv
import os
import google.generativeai as genai

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics


# Create your views here.
def home(request):
    return HttpResponse("Soham is great")




# 1. Get tips for interview
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_interview_tips():
    try:
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(["interview tips"])

        formatted_response = ""
        current_paragraph = ""
        paragraphs = response.text.split("\n\n")  # Split response into paragraphs
        word_count = 0
        for paragraph in paragraphs:
            if word_count + len(paragraph.split()) <= 250:
                current_paragraph += f"{paragraph} "
                word_count += len(paragraph.split())
            else:
                formatted_response += f"{current_paragraph}\n\n"
                current_paragraph = f"{paragraph} "
                word_count = len(paragraph.split())

        # Add the last paragraph if not added already
        if current_paragraph:
            formatted_response += f"{current_paragraph}\n\n"

        return formatted_response
    except Exception as e:
        print(f"Error: {e}")
        return "I'm sorry, I encountered an error and couldn't process your request. Please try again later."
    



class InterviewTips(APIView):

    def post(self, request):

        user_input = request.data['question']
        
        if user_input.lower() == "quit":
            print("Exiting the Interview Tips Bot. Goodbye!")
        
        print("Fetching interview tips...")
        tips = get_interview_tips()
        if tips:
            print("\nHere are some interview tips for you:")
            print(tips)
            return JsonResponse({"Message": tips})
        else:
            return JsonResponse({"Message":"Failed to retrieve interview tips. Please try again later."})



# 2. Prediction for getting placement and recommandations
from rest_framework.response import Response
from rest_framework.views import APIView
from django.http import JsonResponse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

class PlacementPrediction(APIView):

    def post(self, request):
        try:
            # Read data from the request object
            age = int(request.data.get("age"))
            gender = request.data.get("gender")
            stream = request.data.get("stream")
            gpa = float(request.data.get("gpa"))
            internships = int(request.data.get("internships"))

            # Read the CSV file
            df = pd.read_csv('collegePlace.csv')
            
            # Preprocessing
            le = LabelEncoder()
            df['Gender'] = le.fit_transform(df['Gender'])
            df['Stream'] = le.fit_transform(df['Stream'])

            # Define features and target variable
            y = df['PlacedOrNot']
            x = df.drop(columns=['PlacedOrNot','Hostel'])

            # Splitting the data into training and testing sets
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=3)

            # Train Random Forest Classifier
            classifier = RandomForestClassifier(n_estimators=1000, criterion='gini', max_depth=None,
                                                min_samples_split=2, min_samples_leaf=1,
                                                min_weight_fraction_leaf=0.0, max_features='auto',
                                                max_leaf_nodes=None, bootstrap=True, oob_score=False,
                                                n_jobs=1, random_state=None, verbose=0, warm_start=False,
                                                class_weight=None)
            classifier.fit(x_train, y_train)

            # Define a function for prediction
            def predict_placement(Age, Gender, Stream, GPA, Internships):
                # Convert user input for gender to lowercase
                gender_encoded = 1 if Gender.lower() == 'male' else 0
                # Stream input remains unchanged
                stream_encoded = le.transform([Stream])[0]
                data = [[Age, gender_encoded, stream_encoded, GPA, Internships, 0]]  # Add dummy values for extra features
                prediction = classifier.predict(data)
                if prediction == 1:
                    return JsonResponse({'mssg': "Higher chances of getting placement. Keep improving yourself."})
                else:
                    return JsonResponse({'mssg': "Lower chances of getting placement. Focus on improving yourself."})

            # Make prediction
            prediction = predict_placement(age, gender, stream, gpa, internships)
            return prediction

        except FileNotFoundError:
            return JsonResponse({'error': "The file 'collegePlace.csv' does not exist. Please provide the correct file path."})
        
        except PermissionError:
            return JsonResponse({'error': "Permission denied. Make sure you have the necessary permissions to access the file."})

        except Exception as e:
            return JsonResponse({'error': str(e)})



# 3. Mock Interview bot with feedback
        
        