import datetime
import json
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from google.cloud import firestore as gcloud_firestore

# Load the JSON data from the output file
with open("output_file.json", "r") as infile:
    sheets_data = json.load(infile)

# Use the downloaded JSON file to authenticate your Firebase account
cred = credentials.Certificate("desollarfitness-4ca0f-firebase-adminsdk-y0wax-ad2f472b57.json")
firebase_admin.initialize_app(cred)

db = firestore.client()
count = 1
i = 1
# Upload the data to Firebase
for workout in sheets_data["workouts"]:
    print("Processing workout:", workout["emphasis"])
    for group in workout["groups"]:
        week = f'Week {count}'
        if i % 4 == 0:
            count += 1
        print("Processing group:", group["group"], "in", workout["emphasis"], "on", week)
        for exercise in group["exercises"]:
            print("Uploading", exercise["Exercise"], "to", workout["emphasis"], "on", week, "in", group["group"],
                  "group")
            if exercise["Exercise"] == "" or exercise["Exercise"] == "Exercise":
                continue

            server_time = datetime.datetime.now()
            # Convert the date to a string down to the millisecond
            server_time = server_time.strftime("%Y-%m-%d %H:%M:%S.%f")
            # make sure to add data to each collection and document
            db.collection("workouts").document(workout["emphasis"]).set(
                {"emphasis": workout["emphasis"], "time": server_time})
            db.collection("workouts").document(workout["emphasis"]).collection(week).document(group["group"]).set(
                {"group": group["group"]})
            db.collection("workouts").document(workout["emphasis"]).collection(week).document(
                group["group"]).collection(
                "exercises").document(exercise["Exercise"]).set({"Exercise": exercise["Exercise"]})
            # Add the servertime of the upload to the exercise data
            exercise["servertime"] = server_time
            db.collection("workouts").document(workout["emphasis"]).collection(week).document(
                group["group"]).collection("exercises").document(exercise["Exercise"]).set(exercise)
        i += 1

print("Data uploaded to Firebase")
