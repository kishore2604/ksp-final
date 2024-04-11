from flask import Flask, render_template, request, redirect, url_for, session, Response,send_file,make_response,jsonify
from flask_session import Session
from pymongo import MongoClient
import urllib.parse
from new_app import get_row_by_aadhar,create2
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as PlatypusImage
import random
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import PyPDF2
import re
from transformers import pipeline
import cv2
import pandas as pd
import requests
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import numpy as np
from keras.models import load_model
from collections import deque
import os
import base64
import requests
import io
from PIL import Image
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from supervision import Detections



app = Flask(__name__)

app.secret_key = 'abc'
app.config['UPLOAD_FOLDER'] = 'uploads/'
folder_path = "static/output"
mongodb_uri = "mongodb+srv://harshankishore004:" + urllib.parse.quote("harshan@1803") + "@cluster0.n4seoyw.mongodb.net/?retryWrites=true&w=majority"
database_name = 'credentials_db'
client = MongoClient(mongodb_uri)
db = client[database_name]
users_collection = db['users']
API_TOKEN="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"
API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large"
headers = {"Authorization": f"Bearer {API_TOKEN}"}

model_path = hf_hub_download(
    repo_id="arnabdhar/YOLOv8-human-detection-thermal",
    filename="model.pt"
)

model = YOLO(model_path)


API_URL1 = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5"
HEADERS1 = {"Authorization": "Bearer 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"}

model = load_model('modelnew.h5')
Q = deque(maxlen=128)

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

import smtplib

def send_report(email, attachment_path):
    # Email configuration
    sender = 'harshankumarhrk@gmail.com'  
    password = 'kextqzwaumpqzijr'  
    receiver = email

    # Message setup
    message = MIMEMultipart()
    message['From'] = sender
    message['To'] = receiver
    message['Subject'] = "FIR report"

    # Attach PDF file
    with open(attachment_path, 'rb') as file:
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(file.read())
    encoders.encode_base64(part)
    part.add_header('Content-Disposition', f'attachment; filename={attachment_path}')
    message.attach(part)

    # Establish SMTP connection and send email
    with smtplib.SMTP('smtp.gmail.com', 587) as session:
        session.starttls()
        session.login(sender, password)
        session.sendmail(sender, receiver, message.as_string())

    print('Mail Sent')

def query(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
    return response.json()

def generate_pdf_report(form_data):

    doc = SimpleDocTemplate("report.pdf", pagesize=letter)
    styles = getSampleStyleSheet()

    logo = "static/img/logo.png"  
    logo_img = PlatypusImage(logo, width=100, height=100)
    content = []
    content.append(logo_img)
    title = Paragraph("First Information Report", styles['Title'])
    content.append(title)
    for key, value in form_data.items():
        content.append(Paragraph(f"<b>{key}:</b> {value}", styles['Normal']))
        content.append(Spacer(1, 12))  
    doc.build(content)

def generate_pdf_ipc_code(ipc_data):

    doc = SimpleDocTemplate("v_fir.pdf", pagesize=letter)
    styles = getSampleStyleSheet()
    image_path = get_image_path(image_directory)
    logo = image_path 
    logo_img = PlatypusImage(logo, width=100, height=100)
    content = []
    content.append(logo_img)
    title = Paragraph("IPC Virtual FIR", styles['Title'])
    content.append(title)
    content.append(Paragraph(f"<b>{ipc_data}</b>", styles['Normal']))
    # for key, value in form_data.items():
    #     content.append(Paragraph(f"<b>{key}:</b> {value}", styles['Normal']))
    #     content.append(Spacer(1, 12))  
    doc.build(content)

def generate_pdf(fields):
    filename = "output.pdf"
    c = canvas.Canvas(filename, pagesize=letter)
    y = 750
    for field_name, field_value in fields.items():
        c.drawString(100, y, f"{field_name}: {field_value}")
        y -= 20
    c.save()
    return filename

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ''
        for page_num in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page_num].extract_text()
    return text

def extract_text_from_pdf(pdf_path1):
    with open(pdf_path1, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ''
        for page_num in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page_num].extract_text()
    return text

def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def answer_question(context, question):
    qa_pipeline = pipeline('question-answering', model='bert-large-uncased-whole-word-masking-finetuned-squad', tokenizer='bert-large-uncased-whole-word-masking-finetuned-squad')
    result = qa_pipeline(context=context, question=question)
    return result['answer']


def video_processing():
    cap = cv2.VideoCapture(0)
    ret, prev_frame = cap.read()
    c = 0
    displacement_threshold = 20

    while True:
        ret, next_frame = cap.read()

        if not ret:
            break

        frame_diff = cv2.absdiff(prev_frame, next_frame)
        frame_diff_gray = cv2.cvtColor(frame_diff, cv2.COLOR_BGR2RGB)

        mean_diff = cv2.mean(frame_diff_gray)[0]

        if mean_diff > displacement_threshold:
            c += 1
            print(c)
            cv2.imwrite(f'displacement_{c}.jpg', prev_frame)
            # blink_and_sound(0.5, 1.0, 0.1)
            ip=get_ip_address()
            loc=get_location_from_ip(ip)
            print(loc)
            print("Camera displaced")
            
            send_email("harshankumarhrk@gmail.com","displacement_1.jpg")
            # session["img_url"]="displacement_1.jpg"

        else:
            _, buffer = cv2.imencode('.jpg', next_frame)
            frame_data = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        prev_frame = next_frame.copy()

    cap.release()

def get_location_from_ip(ip_address):
    access_token = 'ca2235bc0acff2'  
    url = f'https://ipinfo.io/{ip_address}?token={access_token}'
    
    response = requests.get(url)
    data = response.json()

    city = data.get('city')
    region = data.get('region')
    country = data.get('country')
    loc = data.get('loc') 
    latitude, longitude = loc.split(',') if loc else (None, None)
    location = {
        'city': city,
        'region': region,
        'country': country,
        'latitude': latitude,
        'longitude': longitude
    }

    df = pd.read_csv("police.csv")

  
    given_latitude = float(location["latitude"]) if location["latitude"] else None
    given_longitude = float(location["longitude"]) if location["longitude"] else None

    matching_rows = df[(df["Latitude"] == float(34.9540542)) & (df["Longitude"] == float(135.7515062))]
    
    if not matching_rows.empty:
        police_station_name = matching_rows["Police Station Name"].iloc[0]
        print(police_station_name)
        # send_email("harshankumarhrk@gmail.com",session["img_url"])
        # try:
        #     c=0
        #     while True:
        #         c+=1
        #         board.digital[buzzer_pin].write(1)
        #         time.sleep(0.25)  
        #         board.digital[buzzer_pinpin].write(0)
        #         time.sleep(0.25) 
        #         if(c>2):
        #             break


        # except KeyboardInterrupt:

        #     board.digital[buzzer_pin].write(0)
        #     board.exit()
        # return redirect('email')

    
    location_str = f"{city}, {region}, {country}"
    return location_str

def get_ip_address():
    url = 'https://api.ipify.org'
    response = requests.get(url)
    ip_address = response.text
    return ip_address

def send_email(email, attachment_path):
    
    body = f'''
        Alert needed 
    '''

    sender = 'harshankumarhrk@gmail.com'  
    password = 'kextqzwaumpqzijr'  

    receiver = email

    message = MIMEMultipart()
    message['From'] = sender
    message['To'] = receiver
    message['Subject'] = "Camera Displacement Notification"

    message.attach(MIMEText(body, 'plain'))

    with open(attachment_path, 'rb') as file:
        img_data = file.read()
        image = MIMEImage(img_data, name="case.jpg")
        message.attach(image)

    session = smtplib.SMTP('smtp.gmail.com', 587)
    session.starttls()
    session.login(sender, password)

    # Send the email
    text = message.as_string()
    session.sendmail(sender, receiver, text)

    # Quit the session
    session.quit()
    print('Mail Sent')

def predict_violence(video_path):
    vs = cv2.VideoCapture(video_path)
    violence_frames = []
    frame_count = 0
    while True:
        grabbed, frame = vs.read()
        if not grabbed:
            break

        frame_count += 1
        frame_path = os.path.join(app.config['UPLOAD_FOLDER'], f'frame_{frame_count}.png')
        cv2.imwrite(frame_path, frame)

        frame = cv2.resize(frame, (128, 128)).astype("float32")
        frame = frame.reshape(128, 128, 3) / 255
        preds = model.predict(np.expand_dims(frame, axis=0))[0]
        Q.append(preds)
        results = np.array(Q).mean(axis=0)
        if results > 0.50:
            violence_frames.append(frame_path)

    vs.release()
    return violence_frames

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

image_directory = 'evidence' 

def get_image_path(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                return os.path.relpath(os.path.join(root, file), directory)
            
def query_model(prompt):
    payload = {"inputs": prompt}
    response = requests.post(API_URL1, headers=HEADERS1, json=payload)
    return response.content

def inference(image, output_path):
    model_output = model(image, conf=0.6, verbose=False)
    detections = Detections.from_ultralytics(model_output[0])
    for class_id in detections.class_id:
        if class_id == 0:  
            for bbox in detections.xyxy:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 0), 3)
            cv2.imwrite(output_path, image)
            print("Image saved with bounding boxes marked.")
            return True
    
    return False

@app.route('/')
def start():
    return redirect(url_for('login'))

@app.route('/login', methods=['POST', 'GET'])
def login():
    if request.method == 'POST':
        user_auth = request.form['nm']
        pwd_auth = request.form['pwd']
        user_data = users_collection.find_one({'email': user_auth})

        if user_data and str(user_data['password']) == pwd_auth:
            session['user'] = str(user_data['_id'])  
            return redirect(url_for('home'))
        
    return render_template('login.html')

@app.route('/home', methods=['POST', 'GET'])
def home():
    return render_template('index_2.html')

@app.route('/autofill_aadhar', methods=['POST', 'GET'])
def autofill_aadhar():
    if request.method == 'POST':
        session["aadhar_number"] = request.form.get("aadhar")
        print(session["aadhar_number"])
    
    return render_template('complaint.html')

@app.route('/login_aadhar', methods=['POST', 'GET'])
def login_aadhar():
    
    if request.method == 'POST':
        aadhar_number = request.form.get("aadhar")
        print("Aadhar number received:", aadhar_number)
        session["aadhar_number"] = aadhar_number
        print("Session after setting:", session)
    aadhar_number_session = session.get("aadhar_number")
    print("Aadhar number from session:", aadhar_number_session)
    aadhar_data=get_row_by_aadhar(aadhar_number_session)[0]
    print(aadhar_data[1])
    session["applicant_name"]=aadhar_data[1]
    session["applicant_dob"]=aadhar_data[2]
    session["applicant_gender"]=aadhar_data[3]
    session["applicant_address"]=aadhar_data[4]
    session["applicant_district"]=aadhar_data[5]
    session["applicant_state"]=aadhar_data[6]
    session["applicant_mobile"]=aadhar_data[7]
    session["applicant_mail_id"]=aadhar_data[8]

    return redirect(url_for('get_report'))

@app.route('/get_report', methods=['POST', 'GET'])
def get_report():
    if request.method == 'POST':
        # aadhar_number = request.form.get("aadhar")
        # print("Aadhar number received:", aadhar_number)
        # session["aadhar_number"] = aadhar_number
        # print("Session after setting:", session)
        aadhar_number_session = session.get("aadhar_number")

        crime_type=request.form.get("crime_type")
        session["inp_crime_type"]=crime_type

        date=request.form.get("crime_date")
        session["inp_crime_date"]=date

        time=request.form.get("crime_time")
        session["inp_crime_time"]=time

        place=request.form.get("crime_place")
        session["inp_crime_place"]=place

        victim=request.form.get("crime_victim")
        session["inp_crime_victim"]=victim

        description=request.form.get("crime_description")
        session["inp_crime_description"]=description

        print("aadhar number working",aadhar_number_session)

    return render_template('description_crime.html')

@app.route('/download_report', methods=['POST', 'GET'])
def download_report():
    aadhar_number_session = session.get("aadhar_number")
    crime_type_session=session.get("inp_crime_type")
    date_session=session.get("inp_crime_date")
    time_session=session.get("inp_crime_time")
    crime_place_session=session.get("inp_crime_place")
    victim_session=session.get("inp_crime_victim")
    crime_description_session=session.get("inp_crime_description")
    print("Aadhar number from session:", aadhar_number_session)
    print("crime type",crime_type_session)

    aadhar_data=get_row_by_aadhar(aadhar_number_session)[0]
    print(aadhar_data[1])
    session["applicant_name"]=aadhar_data[1]
    session["applicant_dob"]=aadhar_data[2]
    session["applicant_gender"]=aadhar_data[3]
    session["applicant_address"]=aadhar_data[4]
    session["applicant_district"]=aadhar_data[5]
    session["applicant_state"]=aadhar_data[6]
    session["applicant_mobile"]=aadhar_data[7]
    session["applicant_mail_id"]=aadhar_data[8]

    data_lis=[aadhar_number_session,session["applicant_name"],session["applicant_dob"],session["applicant_gender"],session["applicant_address"],session["applicant_district"],session["applicant_state"],session["applicant_mail_id"],crime_type_session,date_session,time_session,crime_place_session,session["applicant_mobile"],victim_session,crime_description_session]
    data_lis2 = [
            ["Aadhar Number:", aadhar_number_session],
            ["Applicant Name:", session["applicant_name"]],
            ["Date of Birth:", session["applicant_dob"]],
            ["Gender:", session["applicant_gender"]],
            ["Address:", session["applicant_address"]],
            ["District:", session["applicant_district"]],
            ["State:", session["applicant_state"]],
            ["Email ID:", session["applicant_mail_id"]],
            ["Crime Type:", crime_type_session],
            ["Date:", date_session],
            ["Time:", time_session],
            ["Place:", crime_place_session],
            ["Mobile Number:", session["applicant_mobile"]],
            ["Victim:", victim_session],
            ["Crime Description:", crime_description_session]
        ]
    filename = "fir_complaint.pdf"
    doc = SimpleDocTemplate(filename, pagesize=letter)
    table = Table(data_lis2)
    style = TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                            ('GRID', (0, 0), (-1, -1), 1, colors.black)])
    table.setStyle(style)

    doc.build([table])
    if(create2(data_lis)):
        return render_template("download_report.html",pdf_filename=filename)



    
    return render_template('download_report.html')

@app.route('/virtual_fir', methods=['POST', 'GET'])
def fir_details():
    return render_template("fir.html")

@app.route('/submit_report', methods=['POST'])
def submit_report():
    if request.method == 'POST':
        form_data = request.form
        complainant_name = form_data.get('complainantName')
        contact_number = form_data.get('contactNumber')
        address = form_data.get('address')
        aadhar_number = form_data.get('aadharNumber')
        incident_date = form_data.get('incidentDate')
        incident_time = form_data.get('incidentTime')
        incident_location = form_data.get('incidentLocation')
        offence_category = form_data.get('offenceCategory')
        offence_description = form_data.get('offenceDescription')
        crime_description = form_data.get('crimeDescription')
        suspect_information = form_data.get('suspectInformation')

        print(complainant_name)
        print(form_data)  
        

        form_data2 = {
    'Complainant Name': complainant_name,
    'Contact Number': contact_number,
    'Address': address,
    'Aadhar Number': aadhar_number,
    'Incident Date': incident_date,
    'Incident Time': incident_time,
    'Incident Location': incident_location,
    'Offence Category': offence_category,
    'Offence Description': offence_description,
    'Crime Description': crime_description,
    'Suspect Information': suspect_information
}
        generate_pdf_report(form_data)
        send_report("harshankumarhrk@gmail.com","report.pdf")

        return render_template('report_download.html')
    else:
        return 'Method Not Allowed', 405

@app.route('/find_police', methods=['POST', 'GET'])
def find_police():
    return render_template('kn.html')

@app.route('/download_pdf/<filename>')
def download_pdf(filename):
    return send_file(filename, as_attachment=True)

@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')

@app.route('/answer', methods=['POST'])
def get_answer():
    user_question = request.form['question']
    answer = answer_question(cleaned_text, user_question)
    return render_template('chatbot.html', question=user_question, answer=answer)

@app.route('/displacement')
def displacement():
    return render_template('displacement.html')


@app.route('/video_feed')
def video_feed():
    return Response(video_processing(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/classifier', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(video_path)
            violence_frames = predict_violence(video_path)
            return render_template('result.html', violence_frames=violence_frames)
    return render_template('uploader.html')


@app.route('/fir_status_search',methods=['GET', 'POST'])
def fir_status_search():
    
    return render_template("status.html")
@app.route('/aadhar_data_fir',methods=['GET', 'POST'])
def aadhar_data_fir():
    if request.method == 'POST':
        aadhar_number_fir = request.form.get("aadhar_number")
        aadhar_data=get_row_by_aadhar(aadhar_number_fir)[0]
        res=0
        if(aadhar_data[10]=="started"):
            res=1
        elif(aadhar_data[10]=="investigating"):
            res=2
        elif(aadhar_data[10]=="evidence_traced"):
            res=3
        else:
            res=4

        print(res)
        return render_template("progress.html",res=res)
    return "hello"
@app.route('/ipc_fir')
def ipc_fir():
    return render_template('ipc.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = file.filename
        file.save(os.path.join("evidence", filename))
        return redirect(url_for('ipc_get_fir'))
    else:
        return "Invalid file format. Allowed formats are png, jpg, jpeg, gif."
    


@app.route('/ipc_get_fir',methods=['POST','GET'])
def ipc_get_fir():
    image_path = get_image_path(image_directory)
    output=query(image_path)
    query_output=f"under what section this {output} fighting in public  case can be reported "
    # user_question = request.form['question']
    answer = answer_question(cleaned_text2, query_output)
    generate_pdf_ipc_code(answer)
    return answer

@app.route('/evidence_generator')
def evidence_generator():
    return render_template('evidence_bot.html')

@app.route('/get_image', methods=['POST'])
def get_image():
    prompt = request.form['prompt']
    image_bytes = query_model(prompt)
    image = Image.open(io.BytesIO(image_bytes))
    image_path = "static/image.png"
    image.save(image_path)
    return jsonify({'image_path': image_path})

@app.route('/faq',methods=['GET','POST'])
def faq():
    return render_template('faq.html')

@app.route('/tdp',methods=['GET','POST'])
def tdp():
    return render_template('service.html')

@app.route('/resource_allocation',methods=['GET','POST'])
def resource_allocation():
    return render_template('resource.html')

@app.route('/resource_police',methods=['GET','POST'])
def resource_police():
    if 'image' not in request.files:
        return redirect(request.url)
    file = request.files['image']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = file.filename
        file.save(os.path.join("police", filename))
        # Redirect to the page with profile cards
        return redirect(url_for('profile_cards'))
    

@app.route('/profile_cards')
def profile_cards():
    profiles=[
{'name': 'James Davis', 'police_station': 'Station D', 'no_of_stars': 2, 'no_of_cases_handled': 35, 'no_of_cases_solved': 33, 'year_of_experience': 2, 'email': 'jamesdavis@outlook.com', 'contact_number': '0729570670', 'police_id': 'UXWNRSIW','image':'1.jpeg'},
{'name': 'Olivia Taylor', 'police_station': 'Station C', 'no_of_stars': 1, 'no_of_cases_handled': 23, 'no_of_cases_solved': 34, 'year_of_experience': 1, 'email': 'oliviataylor@yahoo.com', 'contact_number': '5382428943', 'police_id': 'Z4L7HVH1','image':'2.jpeg'},
{'name': 'Bob Miller', 'police_station': 'Station A', 'no_of_stars': 5, 'no_of_cases_handled': 63, 'no_of_cases_solved': 35, 'year_of_experience': 3, 'email': 'bobmiller@hotmail.com', 'contact_number': '1473407396', 'police_id': 'N8H97J0K','image':'3.jpeg'},
{'name': 'James Wilson', 'police_station': 'Station A', 'no_of_stars': 1, 'no_of_cases_handled': 72, 'no_of_cases_solved': 30, 'year_of_experience': 1, 'email': 'jameswilson@hotmail.com', 'contact_number': '4403651326', 'police_id': 'H8LWE3LS','image':'4.jpeg'},
{'name': 'Olivia Taylor', 'police_station': 'Station C', 'no_of_stars': 1, 'no_of_cases_handled': 76, 'no_of_cases_solved': 48, 'year_of_experience': 17, 'email': 'oliviataylor@hotmail.com', 'contact_number': '3229966626', 'police_id': '744VZSWK','image':'5.jpeg'},
{'name': 'Bob Jones', 'police_station': 'Station C', 'no_of_stars': 3, 'no_of_cases_handled': 39, 'no_of_cases_solved': 42, 'year_of_experience': 13, 'email': 'bobjones@outlook.com', 'contact_number': '8747558042', 'police_id': 'R0IYCK72','image':'6.jpeg'},
{'name': 'John Jones', 'police_station': 'Station D', 'no_of_stars': 4, 'no_of_cases_handled': 49, 'no_of_cases_solved': 48, 'year_of_experience': 15, 'email': 'johnjones@yahoo.com', 'contact_number': '2689486708', 'police_id': 'KT4ZEAX2','image':'7.jpeg'},
{'name': 'Emily Brown', 'police_station': 'Station A', 'no_of_stars': 4, 'no_of_cases_handled': 28, 'no_of_cases_solved': 42, 'year_of_experience': 5, 'email': 'emilybrown@yahoo.com', 'contact_number': '8543113717', 'police_id': 'EAXPS36W','image':'8.jpeg'},
{'name': 'Sophia Moore', 'police_station': 'Station D', 'no_of_stars': 4, 'no_of_cases_handled': 28, 'no_of_cases_solved': 38, 'year_of_experience': 5, 'email': 'sophiamoore@hotmail.com', 'contact_number': '4701412785', 'police_id': 'S1Y7TVUG','image':'9.jpeg'},
{'name': 'Michael Brown', 'police_station': 'Station C', 'no_of_stars': 4, 'no_of_cases_handled': 99, 'no_of_cases_solved': 49, 'year_of_experience': 11, 'email': 'michaelbrown@outlook.com', 'contact_number': '0567172596', 'police_id': '7GBXHXNI','image':'10.jpeg'},
{'name': 'Alice Taylor', 'police_station': 'Station C', 'no_of_stars': 3, 'no_of_cases_handled': 25, 'no_of_cases_solved': 1, 'year_of_experience': 17, 'email': 'alicetaylor@hotmail.com', 'contact_number': '5910841171', 'police_id': 'Y7MGT9B7','image':'11.jpeg'},
{'name': 'John Taylor', 'police_station': 'Station B', 'no_of_stars': 3, 'no_of_cases_handled': 91, 'no_of_cases_solved': 41, 'year_of_experience': 10, 'email': 'johntaylor@hotmail.com', 'contact_number': '9183756066', 'police_id': 'Z5RSEVIV','image':'12.jpeg'},
{'name': 'Bob Smith', 'police_station': 'Station A', 'no_of_stars': 1, 'no_of_cases_handled': 46, 'no_of_cases_solved': 40, 'year_of_experience': 9, 'email': 'bobsmith@yahoo.com', 'contact_number': '2949065658', 'police_id': 'J5ARH2U5','image':'13.jpeg'},
{'name': 'Emma Brown', 'police_station': 'Station D', 'no_of_stars': 5, 'no_of_cases_handled': 40, 'no_of_cases_solved': 44, 'year_of_experience': 6, 'email': 'emmabrown@outlook.com', 'contact_number': '4345203193', 'police_id': 'D831PPR9','image':'14.jpeg'},
{'name': 'Michael Taylor', 'police_station': 'Station D', 'no_of_stars': 1, 'no_of_cases_handled': 31, 'no_of_cases_solved': 9, 'year_of_experience': 2, 'email': 'michaeltaylor@outlook.com', 'contact_number': '9859690302', 'police_id': 'X4OYLZSH','image':'15.jpeg'},
{'name': 'David Jones', 'police_station': 'Station B', 'no_of_stars': 5, 'no_of_cases_handled': 16, 'no_of_cases_solved': 20, 'year_of_experience': 5, 'email': 'davidjones@yahoo.com', 'contact_number': '9361006581', 'police_id': '5CJPXWM1','image':'16.jpeg'},
{'name': 'Emma Williams', 'police_station': 'Station B', 'no_of_stars': 4, 'no_of_cases_handled': 90, 'no_of_cases_solved': 18, 'year_of_experience': 18, 'email': 'emmawilliams@yahoo.com', 'contact_number': '6123627902', 'police_id': 'TQU4MAAU','image':'17.jpeg'},
{'name': 'Emma Davis', 'police_station': 'Station B', 'no_of_stars': 5, 'no_of_cases_handled': 92, 'no_of_cases_solved': 18, 'year_of_experience': 6, 'email': 'emmadavis@hotmail.com', 'contact_number': '1127250266', 'police_id': 'L4LJR25O','image':'18.jpeg'},
{'name': 'Emma Davis', 'police_station': 'Station D', 'no_of_stars': 1, 'no_of_cases_handled': 100, 'no_of_cases_solved': 42, 'year_of_experience': 5, 'email': 'emmadavis@outlook.com', 'contact_number': '5323147214', 'police_id': '53LV8L6G','image':'19.jpeg'},
{'name': 'Michael Brown', 'police_station': 'Station D', 'no_of_stars': 2, 'no_of_cases_handled': 59, 'no_of_cases_solved': 6, 'year_of_experience': 16, 'email': 'michaelbrown@outlook.com', 'contact_number': '1524462527', 'police_id': 'EAMQCKNQ','image':'20.jpeg'},
{'name': 'Emma Williams', 'police_station': 'Station C', 'no_of_stars': 2, 'no_of_cases_handled': 30, 'no_of_cases_solved': 23, 'year_of_experience': 18, 'email': 'emmawilliams@gmail.com', 'contact_number': '0663926876', 'police_id': 'DV2T7QYJ','image':'21.jpeg'},
{'name': 'Bob Johnson', 'police_station': 'Station A', 'no_of_stars': 2, 'no_of_cases_handled': 10, 'no_of_cases_solved': 3, 'year_of_experience': 16, 'email': 'bobjohnson@hotmail.com', 'contact_number': '4874142179', 'police_id': 'R4WQRM0J','image':'22.jpeg'},
{'name': 'David Moore', 'police_station': 'Station D', 'no_of_stars': 2, 'no_of_cases_handled': 10, 'no_of_cases_solved': 2, 'year_of_experience': 10, 'email': 'davidmoore@outlook.com', 'contact_number': '0969096222', 'police_id': 'WUV7030J','image':'23.jpeg'},
{'name': 'Sophia Brown', 'police_station': 'Station D', 'no_of_stars': 2, 'no_of_cases_handled': 53, 'no_of_cases_solved': 23, 'year_of_experience': 14, 'email': 'sophiabrown@yahoo.com', 'contact_number': '0946995703', 'police_id': '909QXUJ3','image':'24.jpeg'},
{'name': 'Bob Johnson', 'police_station': 'Station C', 'no_of_stars': 5, 'no_of_cases_handled': 66, 'no_of_cases_solved': 29, 'year_of_experience': 14, 'email': 'bobjohnson@gmail.com', 'contact_number': '8797123627', 'police_id': 'S7I85QLT','image':'25.jpeg'},
{'name': 'David Davis', 'police_station': 'Station A', 'no_of_stars': 1, 'no_of_cases_handled': 56, 'no_of_cases_solved': 31, 'year_of_experience': 5, 'email': 'daviddavis@yahoo.com', 'contact_number': '0231466744', 'police_id': 'I5VNGVQS','image':'26.jpeg'},
{'name': 'James Smith', 'police_station': 'Station A', 'no_of_stars': 5, 'no_of_cases_handled': 52, 'no_of_cases_solved': 7, 'year_of_experience': 17, 'email': 'jamessmith@gmail.com', 'contact_number': '0666650728', 'police_id': '0EIXM4LX','image':'27.jpeg'},
{'name': 'Emma Miller', 'police_station': 'Station A', 'no_of_stars': 5, 'no_of_cases_handled': 96, 'no_of_cases_solved': 43, 'year_of_experience': 12, 'email': 'emmamiller@yahoo.com', 'contact_number': '4860795731', 'police_id': 'J765P81N','image':'28.jpeg'},
{'name': 'Emma Williams', 'police_station': 'Station D', 'no_of_stars': 5, 'no_of_cases_handled': 14, 'no_of_cases_solved': 3, 'year_of_experience': 12, 'email': 'emmawilliams@hotmail.com', 'contact_number': '3754636947', 'police_id': 'YWFBC8X4','image':'29.jpeg'},
{'name': 'Sophia Brown', 'police_station': 'Station C', 'no_of_stars': 3, 'no_of_cases_handled': 69, 'no_of_cases_solved': 46, 'year_of_experience': 1, 'email': 'sophiabrown@outlook.com', 'contact_number': '0076972551', 'police_id': 'ISKA80YC','image':'30.jpeg'},
{'name': 'Alice Miller', 'police_station': 'Station C', 'no_of_stars': 1, 'no_of_cases_handled': 16, 'no_of_cases_solved': 3, 'year_of_experience': 17, 'email': 'alicemiller@gmail.com', 'contact_number': '5453612245', 'police_id': 'RWD3ICZB','image':'31.jpeg'},
{'name': 'Emma Miller', 'police_station': 'Station C', 'no_of_stars': 3, 'no_of_cases_handled': 45, 'no_of_cases_solved': 49, 'year_of_experience': 12, 'email': 'emmamiller@yahoo.com', 'contact_number': '7481076365', 'police_id': '4AQH6BZB','image':'32.jpeg'},
{'name': 'James Miller', 'police_station': 'Station B', 'no_of_stars': 4, 'no_of_cases_handled': 36, 'no_of_cases_solved': 6, 'year_of_experience': 3, 'email': 'jamesmiller@outlook.com', 'contact_number': '3246317328', 'police_id': 'QFG037WL','image':'33.jpeg'},
{'name': 'Bob Smith', 'police_station': 'Station C', 'no_of_stars': 2, 'no_of_cases_handled': 57, 'no_of_cases_solved': 9, 'year_of_experience': 8, 'email': 'bobsmith@gmail.com', 'contact_number': '8743320538', 'police_id': 'A7V5PWYT','image':'34.jpeg'},
{'name': 'Alice Miller', 'police_station': 'Station D', 'no_of_stars': 1, 'no_of_cases_handled': 85, 'no_of_cases_solved': 10, 'year_of_experience': 10, 'email': 'alicemiller@gmail.com', 'contact_number': '7037236938', 'police_id': 'EEO4L1OO','image':'35.jpeg'},
{'name': 'Olivia Johnson', 'police_station': 'Station D', 'no_of_stars': 5, 'no_of_cases_handled': 20, 'no_of_cases_solved': 29, 'year_of_experience': 13, 'email': 'oliviajohnson@gmail.com', 'contact_number': '1292225994', 'police_id': 'J4TV66KI','image':'36.jpeg'},
{'name': 'Alice Jones', 'police_station': 'Station D', 'no_of_stars': 5, 'no_of_cases_handled': 71, 'no_of_cases_solved': 10, 'year_of_experience': 6, 'email': 'alicejones@yahoo.com', 'contact_number': '4732517454', 'police_id': 'X121JPP0','image':'37.jpeg'},
{'name': 'Olivia Taylor', 'police_station': 'Station C', 'no_of_stars': 1, 'no_of_cases_handled': 95, 'no_of_cases_solved': 25, 'year_of_experience': 12, 'email': 'oliviataylor@yahoo.com', 'contact_number': '3750758982', 'police_id': 'YDZI2DC0','image':'38.jpeg'},
{'name': 'Olivia Moore', 'police_station': 'Station D', 'no_of_stars': 2, 'no_of_cases_handled': 86, 'no_of_cases_solved': 27, 'year_of_experience': 6, 'email': 'oliviamoore@outlook.com', 'contact_number': '5718287557', 'police_id': 'YJC6T607','image':'39.jpeg'},
{'name': 'Bob Taylor', 'police_station': 'Station A', 'no_of_stars': 2, 'no_of_cases_handled': 83, 'no_of_cases_solved': 23, 'year_of_experience': 15, 'email': 'bobtaylor@hotmail.com', 'contact_number': '9079347100', 'police_id': 'PX3RLIHZ','image':'40.jpeg'},
{'name': 'Alice Moore', 'police_station': 'Station C', 'no_of_stars': 3, 'no_of_cases_handled': 44, 'no_of_cases_solved': 14, 'year_of_experience': 19, 'email': 'alicemoore@outlook.com', 'contact_number': '1221666023', 'police_id': 'R9YNX9HE','image':'41.jpeg'},
{'name': 'David Johnson', 'police_station': 'Station D', 'no_of_stars': 1, 'no_of_cases_handled': 71, 'no_of_cases_solved': 42, 'year_of_experience': 11, 'email': 'davidjohnson@outlook.com', 'contact_number': '3173815134', 'police_id': 'EQN3974R','image':'42.jpeg'},
{'name': 'Olivia Johnson', 'police_station': 'Station C', 'no_of_stars': 1, 'no_of_cases_handled': 99, 'no_of_cases_solved': 14, 'year_of_experience': 7, 'email': 'oliviajohnson@outlook.com', 'contact_number': '5684585878', 'police_id': '2Y4PVN93','image':'43.jpeg'},
{'name': 'Bob Miller', 'police_station': 'Station D', 'no_of_stars': 2, 'no_of_cases_handled': 16, 'no_of_cases_solved': 2, 'year_of_experience': 4, 'email': 'bobmiller@gmail.com', 'contact_number': '9198278325', 'police_id': 'P8ROX6FY','image':'44.jpeg'},
{'name': 'John Moore', 'police_station': 'Station C', 'no_of_stars': 4, 'no_of_cases_handled': 80, 'no_of_cases_solved': 33, 'year_of_experience': 7, 'email': 'johnmoore@hotmail.com', 'contact_number': '5329915349', 'police_id': 'V8X195KN','image':'45.jpeg'},
{'name': 'Michael Moore', 'police_station': 'Station D', 'no_of_stars': 4, 'no_of_cases_handled': 29, 'no_of_cases_solved': 9, 'year_of_experience': 15, 'email': 'michaelmoore@outlook.com', 'contact_number': '6556433992', 'police_id': 'ZOB11IQM','image':'46.jpeg'},
{'name': 'Emily Smith', 'police_station': 'Station D', 'no_of_stars': 2, 'no_of_cases_handled': 42, 'no_of_cases_solved': 45, 'year_of_experience': 10, 'email': 'emilysmith@gmail.com', 'contact_number': '8878111790', 'police_id': 'RP0BB3L3','image':'47.jpeg'},
{'name': 'Alice Taylor', 'police_station': 'Station C', 'no_of_stars': 5, 'no_of_cases_handled': 65, 'no_of_cases_solved': 36, 'year_of_experience': 1, 'email': 'alicetaylor@hotmail.com', 'contact_number': '9164463059', 'police_id': 'G79631E6','image':'48.jpeg'},
{'name': 'Alice Taylor', 'police_station': 'Station D', 'no_of_stars': 3, 'no_of_cases_handled': 70, 'no_of_cases_solved': 8, 'year_of_experience': 13, 'email': 'alicetaylor@gmail.com', 'contact_number': '8759043612', 'police_id': 'OYOS1HVX','image':'49.jpeg'},
{'name': 'Emma Williams', 'police_station': 'Station B', 'no_of_stars': 4, 'no_of_cases_handled': 85, 'no_of_cases_solved': 6, 'year_of_experience': 4, 'email': 'emmawilliams@yahoo.com', 'contact_number': '1696965175', 'police_id': 'DFXJV2VL','image':'50.jpeg'}
]
    random_profiles = random.sample(profiles, 3)
    return render_template('profile.html', profiles=random_profiles)

if __name__ == "__main__":
    pdf_path = 'ksp.pdf'
    pdf_text = extract_text_from_pdf(pdf_path)
    pdf_path1 = 'act.pdf'
    pdf_text1 = extract_text_from_pdf(pdf_path1)
    
    cleaned_text = preprocess_text(pdf_text)
    cleaned_text2 = preprocess_text(pdf_text1)
    app.run(host='0.0.0.0', port=8000, debug=True)
