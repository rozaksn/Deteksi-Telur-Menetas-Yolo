from ultralytics import YOLO
import firebase_admin
from firebase_admin import credentials, db

model = YOLO('eggs_200_32.pt')

result = model(source=0, show=True, save=False, stream=True)

# File json untuk autentifikasi dan url untuk menghubungkan ke Firebase
database_url = 'https://inkubator-11a6d-default-rtdb.firebaseio.com/'
serviceAccountKeyFile = 'key.json'

cred = credentials.Certificate(serviceAccountKeyFile)
firebase_admin.initialize_app(cred, {
    'databaseURL': database_url
})

db_ref = db.reference('detection')
def check_connection():
    try:
        # Mengecek koneksi dengan  Firebase Realtime Database
        db.reference('/').get()
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

# Check the connection
print(check_connection())

def send_to_firebase(names, confidence):
    data = {
     
        'object_name': names,
        'confidence': confidence
    }
    db_ref.set(data)  
   
for r in result:
    boxes = r.boxes
    for box in boxes:
        object_name = r.names[int(box.cls[0])]
        confidence = float(box.conf)
        

        send_to_firebase(object_name, confidence)
