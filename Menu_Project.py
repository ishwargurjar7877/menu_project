import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from twilio.rest import Client  # Ensure Twilio library is installed
from bs4 import BeautifulSoup
import requests
import geocoder
from gtts import gTTS
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pycaw.pycaw import AudioUtilities, ISimpleAudioVolume

def send_email():
    sender_email = "--"  # Replace with your email address
    receiver_email = input("Enter the receiver's email address: ")
    password = input("Type your password or app password and press enter: ")

    subject = "Test Email from Python"
    body = "This is a test email sent from a Python script."

    # Create a multipart message and set headers
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = subject

    # Add body to email
    message.attach(MIMEText(body, "plain"))

    # Create a secure SSL context
    context = ssl.create_default_context()

    try:
        # Connect to the server and send the email
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, message.as_string())
        print("Email sent successfully!")
    except smtplib.SMTPAuthenticationError as e:
        print(f"Authentication Error: {e}")
    except Exception as e:
        print(f"Error: {e}")

def send_sms():
    account_sid = '--'  # Replace with your Twilio account SID
    auth_token = '--'  # Replace with your Twilio auth token
    client = Client(account_sid, auth_token)

    from_number = '--'  # Replace with your Twilio number
    to_number = input("Enter the recipient's phone number: ")
    message_body = input("Enter the message to send: ")

    message = client.messages.create(
        body=message_body,
        from_=from_number,
        to=to_number
    )

    print(f"Message sent! SID: {message.sid}")

def scrape_google():
    query = input("Enter the search query: ")
    query = query.replace(' ', '+')
    url = f"https://www.google.com/search?q={query}"

    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    results = soup.find_all('h3')
    for i, result in enumerate(results[:5], start=1):
        print(f"{i}. {result.get_text()}")

def find_geo_coordinates():
    g = geocoder.ip('me')
    print(f"Current location: {g.latlng}")

def text_to_audio():
    text = input("Enter the text to convert to audio: ")
    tts = gTTS(text=text, lang='en')
    tts.save("output.mp3")
    os.system("start output.mp3")  # This command works on Windows
    print("Audio saved as output.mp3 and is now playing.")

def control_volume():
    try:
        volume = input("Enter the volume level (0-100): ")
        volume = int(volume)
        
        if 0 <= volume <= 100:
            sessions = AudioUtilities.GetAllSessions()
            for session in sessions:
                volume_control = session._ctl.QueryInterface(ISimpleAudioVolume)
                volume_control.SetMasterVolume(float(volume) / 100, None)
            
            print(f"Volume set to {volume}%")
        else:
            print("Volume must be between 0 and 100.")
    
    except ValueError:
        print("Invalid volume level.")

def connect_mobile_send_sms():
    phone_number = input("Enter the recipient's phone number: ")
    message = input("Enter the message: ")
    os.system(f'adb shell am start -a android.intent.action.SENDTO -d sms:{phone_number} --es sms_body "{message}" --ez exit_on_sent true')
    os.system('adb shell input keyevent 22')
    os.system('adb shell input keyevent 66')
    print("SMS sent from mobile.")

def send_bulk_email():
    sender_email = "--"  # Replace with your email address
    password = input("Type your password or app password and press enter: ")

    subject = "Bulk Email from Python"
    body = "This is a bulk email sent from a Python script."

    receiver_emails = input("Enter the recipient email addresses separated by commas: ").split(',')

    # Create a secure SSL context
    context = ssl.create_default_context()

    for receiver_email in receiver_emails:
        # Create a multipart message and set headers
        message = MIMEMultipart()
        message["From"] = sender_email
        message["To"] = receiver_email.strip()
        message["Subject"] = subject
        message.attach(MIMEText(body, "plain"))

        try:
            with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
                server.login(sender_email, password)
                server.sendmail(sender_email, receiver_email.strip(), message.as_string())
            print(f"Email sent to {receiver_email.strip()}!")
        except smtplib.SMTPAuthenticationError as e:
            print(f"Authentication Error: {e}")
        except Exception as e:
            print(f"Error: {e}")

def data_processing():
    file_path = input("Enter the path to the CSV file: ")

    try:
        df = pd.read_csv(file_path)
        print("Data before processing:")
        print(df.head())

        scaler = StandardScaler()
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

        print("Data after processing:")
        print(df.head())
    except Exception as e:
        print(f"Error: {e}")


def crop_face_from_image():
    file_path = input("Enter the path to the image file: ")
    image = cv2.imread(file_path)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        face = image[y:y+h, x:x+w]
        cv2.imshow('Face', face)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    cv2.imwrite("face_crop.jpg", face)
    print("Face cropped and saved as face_crop.jpg.")

def apply_image_filters():
    file_path = input("Enter the path to the image file: ")
    image = cv2.imread(file_path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(image, (15, 15), 0)
    edges = cv2.Canny(image, 100, 200)

    cv2.imshow('Original', image)
    cv2.imshow('Gray', gray)
    cv2.imshow('Blurred', blurred)
    cv2.imshow('Edges', edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite("gray.jpg", gray)
    cv2.imwrite("blurred.jpg", blurred)
    cv2.imwrite("edges.jpg", edges)
    print("Filters applied and images saved.")

def create_custom_image():
    # Create an image with random colors
    image = np.random.rand(100, 100, 3)

    plt.imshow(image)
    plt.title("Custom Image")
    plt.show()

    plt.imsave("custom_image.png", image)
    print("Custom image saved as custom_image.png.")

def click_image_apply_filters():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot access the camera.")
        return

    print("Press 's' to save the image and apply filters, 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        cv2.imshow('Live Feed', frame)

        key = cv2.waitKey(1)
        if key == ord('s'):
            cv2.imwrite("captured_image.jpg", frame)
            print("Image captured and saved as captured_image.jpg.")
            break
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Apply filters to the captured image
    image = cv2.imread("captured_image.jpg")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(image, (15, 15), 0)
    edges = cv2.Canny(image, 100, 200)

    cv2.imshow('Original', image)
    cv2.imshow('Gray', gray)
    cv2.imshow('Blurred', blurred)
    cv2.imshow('Edges', edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite("captured_gray.jpg", gray)
    cv2.imwrite("captured_blurred.jpg", blurred)
    cv2.imwrite("captured_edges.jpg", edges)
    print("Filters applied to captured image and saved.")

def python_tasks_menu():
    print("Select a Python task:")
    print("1. Send Email")
    print("2. Send SMS")
    print("3. Scrape Google")
    print("4. Find Geo Coordinates")
    print("5. Text-to-Audio Conversion")
    print("6. Control Volume")
    print("7. Connect Mobile and Send SMS")
    print("8. Send Bulk Email")
    
    choice = input("Enter your choice: ")
    
    if choice == '1':
        send_email()
    elif choice == '2':
        send_sms()
    elif choice == '3':
        scrape_google()
    elif choice == '4':
        find_geo_coordinates()
    elif choice == '5':
        text_to_audio()
    elif choice == '6':
        control_volume()
    elif choice == '7':
        connect_mobile_send_sms()
    elif choice == '8':
        send_bulk_email()
    else:
        print("Invalid choice. Please try again.")

def ml_tasks_menu():
    print("Select a Machine Learning task:")
    print("1. Data Processing")
    print("2. Crop Face from Image")
    print("3. Apply Image Filters")
    print("4. Create Custom Image")
    print("5. Click Image and Apply Filters")
    
    choice = input("Enter your choice: ")
    
    if choice == '1':
        data_processing()
    elif choice == '2':
        crop_face_from_image()
    elif choice == '3':
        apply_image_filters()
    elif choice == '4':
        create_custom_image()
    elif choice == '5':
        click_image_apply_filters()
    else:
        print("Invalid choice. Please try again.")

def main_menu():
    print("Select a category:")
    print("1. Python Tasks")
    print("2. Machine Learning Tasks")
    
    choice = input("Enter your choice: ")
    
    if choice == '1':
        python_tasks_menu()
    elif choice == '2':
        ml_tasks_menu()
    else:
        print("Invalid choice. Please try again.")

if __name__ == "__main__":
    while True:
        main_menu()
        cont = input("Do you want to perform another task? (yes/no): ")
        if cont.lower() != 'yes':
            break
