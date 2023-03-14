import face_recognition
import cv2
import numpy as np
import csv
import os
from datetime import datetime

video_cap = cv2.VideoCapture(0)

jobs_img = face_recognition.load_image_file("photos/jobs.jpg")
jobs_encoding = face_recognition.face_encodings(jobs_img)[0]

omkar_mhetre_img = face_recognition.load_image_file("photos/omkar.jpg")
omkar_mhetre_encodings = face_recognition.face_encodings(omkar_mhetre_img)[0]

known_face_encoding = [
    jobs_encoding,
    omkar_mhetre_encodings
]

known_face_names = [
    "steve jobs",
    "Omkar Mhetre"
]

students = known_face_names.copy()

face_locations = []
face_encodings = []
face_names = []
s=True

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")


f = open(current_date+'.csv','w+',newline='')
lnwriter = csv.writer(f)

while True:
    _,frame= video_cap.read()
    small_frame = cv2.resize(frame,(0,0),fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:,:,::-1]
    if s:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame,face_locations)
        face_names = []
        for face_encodings in face_encodings:
            matches= face_recognition.compare_faces(known_face_encoding,face_encodings)
            name = ""
            face_distance = face_recognition.face_distance(known_face_encoding,face_encodings)
            best_match_index = np.argmin(face_distance)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)
            if name in known_face_names:
                if name in students:
                    students.remove(name)
                    print(students)
                    current_time = now.strftime("%H-%M-%S")
                    lnwriter.writerow([name,current_time])

    cv2.imshow("attendence system",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_cap.release()
cv2.destroyAllWindows()
f.close()