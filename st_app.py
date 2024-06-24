# Scan
import streamlit as st
import numpy as np
import requests
import tensorflow as tf
from PIL import Image
from tensorflow.keras.utils import img_to_array, load_img

# # ______________________________________________
s = """Apple Pie: ~2.5 calories per gram
Baby Back Ribs: ~3.5 calories per gram
Baklava: ~5 calories per gram
Beef Carpaccio: ~2 calories per gram
Beef Tartare: ~2.5 calories per gram
Beet Salad: ~0.5 calories per gram
Beignets: ~3.5 calories per gram
Bibimbap: ~1.5 calories per gram
Bread Pudding: ~2.5 calories per gram
Breakfast Burrito: ~2 calories per gram
Bruschetta: ~1 calorie per gram
Caesar Salad: ~0.5 calories per gram
Cannoli: ~3.5 calories per gram
Caprese Salad: ~1 calorie per gram
Carrot Cake: ~3.5 calories per gram
Ceviche: ~0.5 calories per gram
Cheese Plate: ~3.5 calories per gram
Cheesecake: ~3.5 calories per gram
Chicken Curry: ~1.5 calories per gram
Chicken Quesadilla: ~2.5 calories per gram
Chicken Wings: ~3 calories per gram
Chocolate Cake: ~4 calories per gram
Chocolate Mousse: ~3 calories per gram
Churros: ~4 calories per gram
Clam Chowder: ~1.5 calories per gram
Club Sandwich: ~2.5 calories per gram
Crab Cakes: ~2 calories per gram
Creme Brulee: ~3.5 calories per gram
Croque Madame: ~3 calories per gram
Cupcakes: ~3.5 calories per gram
Deviled Eggs: ~1 calorie per gram
Donuts: ~4 calories per gram
Dumplings: ~2.5 calories per gram
Edamame: ~1 calorie per gram
Eggs Benedict: ~2.5 calories per gram
Escargots: ~1 calorie per gram
Falafel: ~2 calories per gram
Filet Mignon: ~2.5 calories per gram
Fish and Chips: ~2.5 calories per gram
Foie Gras: ~4.5 calories per gram
French Fries: ~3.5 calories per gram
French Onion Soup: ~1 calorie per gram
French Toast: ~2 calories per gram
Fried Calamari: ~2.5 calories per gram
Fried Rice: ~1.5 calories per gram
Frozen Yogurt: ~1 calorie per gram
Garlic Bread: ~4 calories per gram
Gnocchi: ~1.5 calories per gram
Greek Salad: ~0.5 calories per gram
Grilled Cheese Sandwich: ~3 calories per gram
Grilled Salmon: ~2 calories per gram
Guacamole: ~2 calories per gram
Gyoza: ~2 calories per gram
Hamburger: ~3.5 calories per gram
Hot and Sour Soup: ~0.5 calories per gram
Hot Dog: ~3.5 calories per gram
Huevos Rancheros: ~2 calories per gram
Hummus: ~1.5 calories per gram
Ice Cream: ~2 calories per gram
Lasagna: ~1.5 calories per gram
Lobster Bisque: ~1 calorie per gram
Lobster Roll Sandwich: ~2.5 calories per gram
Macaroni and Cheese: ~3 calories per gram
Macarons: ~4 calories per gram
Miso Soup: ~0.5 calories per gram
Mussels: ~0.5 calories per gram
Nachos: ~2.5 calories per gram
Omelette: ~1.5 calories per gram
Onion Rings: ~2.5 calories per gram
Oysters: ~0.5 calories per gram
Pad Thai: ~2 calories per gram
Paella: ~1.5 calories per gram
Pancakes: ~2 calories per gram
Panna Cotta: ~3.5 calories per gram
Peking Duck: ~4 calories per gram
Pho: ~1 calorie per gram
Pizza: ~2.5 calories per gram
Pork Chop: ~2.5 calories per gram
Poutine: ~2.5 calories per gram
Prime Rib: ~2.5 calories per gram
Pulled Pork Sandwich: ~2.5 calories per gram
Ramen: ~1 calorie per gram
Ravioli: ~1.5 calories per gram
Red Velvet Cake: ~4 calories per gram
Risotto: ~1.5 calories per gram
Samosa: ~2 calories per gram
Sashimi: ~1 calorie per gram
Scallops: ~1 calorie per gram
Seaweed Salad: ~0.5 calories per gram
Shrimp and Grits: ~2 calories per gram
Spaghetti Bolognese: ~1.5 calories per gram
Spaghetti Carbonara: ~2 calories per gram
Spring Rolls: ~1.5 calories per gram
Steak: ~2.5 calories per gram
Strawberry Shortcake: ~3.5 calories per gram
Sushi: ~1 calorie per gram
Tacos: ~2 calories per gram
Takoyaki: ~2.5 calories per gram
Tiramisu: ~3 calories per gram
Tuna Tartare: ~1.5 calories per gram
Waffles: ~2 calories per gram
"""
calories = s.splitlines()
s = "These values are approximations and can vary based on factors such as ingredients and cooking methods."

def rescale(img):
    img2 = img_to_array(img)
    img2 = img2/255
    img2 = img2.reshape(1,224,224,3)
    return img2

# def line():
#     print("_"*40)

def predict(model,img):
    input_details = model.get_input_details()
    output_details = model.get_output_details()
    model.set_tensor(input_details[0]['index'],img)
    model.invoke()
    output_data = model.get_tensor(output_details[0]['index'])
    return output_data

# def detect(original_img):
#   output = ""
#   img = rescale(original_img)
#   # print("-> image recognition...")
#   # p1 = model1.predict(img).argmax()
#   p1 = predict(model1,img).argmax()
#   # Food
#   if p1==0:
#     output = output+"Food image detected!\n"
#     # print("Food image detected!")
#     # print("-> Calories detection...")
#     p1 = predict(model16,img).argmax()
#     # print(calories[p1],'\nNote:',s)
#     output = output +f"{calories[p1]}'\nNote:',{s}\n"
#   # Medical Imaging
#   elif p1==1:
#     # print("Medical Imaging image detected!")
#     output = output+"Medical Imaging image detected!\n"
#     # print("-> Scan type detection...")
#     p1 = predict(model2,img).argmax()
# #_____________________________________________________________ 
#     # CT
#     if p1==0:
#       output = output+"CT Scan detected!"
#       # print("CT Scan detected!")
#       # print("-> Diagnosis detection...")
#       p1 = predict(model14,img).argmax()
#       # print(ct[p1],"detected!")
#       output = output+ f"{ct[p1]} detected!\n"
#       if ct[p1] == "Normal":
#         pass
#       else:
#         # line()
#         # print(scan_info[ct[p1]])
#         output = output +f"{scan_info[ct[p1]]}\n"
#         # line()
#     # MRI
#     elif p1==1:
#       # print("MRI detected!")
#       output = output+"MRI detected!\n"
#       p1 = (predict(model8,img)>=0.5).astype(int)[0,0]
#       # Brain MRI
#       if p1==0:
#         output = output+"Brain MRI detected!\n"
#         # print("Brain MRI detected!")
#         # print("-> Tumors detection...")
#         p1 = (predict(model9,img)>=0.5).astype(int)[0,0]
#         # No tumor
#         if p1==0:
#           output = output+"No tumor found\n"
#           # print("No tumor found")
#         # tumor
#         else: # =1
#           output =output+"Brain tumor detected!\n"
#           # print("Brain tumor detected!")
#           # print("-> Tumor type detection...")
#           p1 = predict(model10,img).argmax()
#           # print(tumor_type[p1],"type detected!")
#           output = output+f"{tumor_type[p1]} type detected!\n"
#           if tumor_type[p1] == "normal no_tumor":
#             pass
#           else:
#             # line()
#             # print(scan_info[tumor_type[p1]])
#             output = output+f"{scan_info[tumor_type[p1]]}\n"
#             # line()
#       # Breast MRI
#       else: # =1
#         output = output+"Breast MRI detected!\n"
#         # print("Breast MRI detected!")
#         # print("-> Breast Cancer detection...")
#         p1 = (predict(model12,img)>=0.5).astype(int)[0,0]
#         # no cancer
#         if p1==0:
#           output = output+"Healthy breast\n"
#           # print("Healthy breast")
#         # Breast cancer
#         else: #1
#           output = output+"Breast Cancer detected!\n"
#           # print("Breast Cancer detected!")
#           # print("-> Cancer type detection...")
#           p1 = (predict(model13,img)>=0.5).astype(int)[0,0]
#           # Benign Breast Cancer
#           if p1==0:
#             output = output+"Benign breast cancer detected!\n"
#             # print("Benign breast cancer detected!")
#           # Malignant
#           else: #1
#             # print("Malignant breast cancer detected!")
#             output = output+"Malignant breast cancer detected!\n"
#           # line()
#           output = output+f"{scan_info['breast_cancer']}\n"
#           # print(scan_info["breast_cancer"])
#           # line()
#     # OCT
#     elif p1==2:
#       output = output+"OCT Scan detected!\n"
#       # print("OCT Scan detected!")
#       # print("-> Diseases detection...")
#       p1 = predict(model15,img).argmax()
#       # print(oct[p1],"eyes detected!")
#       output = output+f"{oct[p1]} eyes detected!\n"
#       if oct[p1] == "Normal":
#         pass
#       else:
#         # line()
#         output = output+f"{scan_info[oct[p1]]}\n"
#         print(scan_info[oct[p1]])
#         # line()
#     # xray
#     else: # = 3
#       output = output+"x-ray detected!\n"
#       # print("x-ray detected!")
#       # print("->Anatomical recognition...")
#       p1 = predict(model3,img).argmax()
#       # chest body part
#       if p1==0:
#         output = output+"Chest x-ray detected!\n"
#         # print("Chest x-ray detected!")
#         # print("->Covid detection...")
#         p1 = (predict(model4,img)).argmax()
#         # covid
#         if p1==0:
#           output = output+"Covid_19 detected!\n"
#           # print("Covid_19 detected!")
#           # line()
#           output = output+f"{scan_info['Covid']}\n"
#           # print(scan_info["Covid"])
#           # line()
#         # Normal
#         elif p1==1:
#           output = output+"Normal Healthy chest x-ray\n"
#           # print("Normal Healthy chest x-ray")
#         # Pneumonia
#         elif p1==2:
#           output = output+"Pneumonia on chest detected!\n"
#           # print("Pneumonia on chest detected!")
#           # line()
#           # print(scan_info["Pneumonia"])
#           output = output+f"{scan_info['Pneumonia']}\n"
#           # line()
#       # Other body parts
#       else:
#         # print(body_parts[p1],"body part x-ray detected!")
#         output = output+f"{body_parts[p1]} body part x-ray detected!\n"
#         # Bone fracture detection
#         # print("-> Bone fracture detection...")
#         p1 = (predict(model5,img)>=0.5).astype(int)[0,0]
#         # Fractured
#         if p1==0:
#           output = output+"Fractured bones detected!\n"
#           # print("Fracture bone detected!")
#           # print("-> Fracture type detection...")
#           p1 = (predict(model6,img)).argmax(axis=1)[0]
#           # print(fracture_type[p1])
#           output = output+f"{fracture_type[p1]}\n"
#           # line()
#           # print(scan_info[fracture_type[p1]])
#           output = output +f"{scan_info[fracture_type[p1]]}"
#           # line()
#         # Not fractured
#         else: # =1
#           # print("Normal unfractured bones")
#           output = output+"Normal unfractured bones\n"
# #_____________________________________________________________
#   # Other
#   else: # == 2
#       # print("ERROR: Unsupported image object!\nPlease try to enter a valid 'Medical imaging' or 'food' image")
#       output = output+"ERROR: Unsupported image object!\nPlease try to enter a valid 'Medical imaging' or 'food' image\n"
#   return output

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

url = "https://drive.google.com/uc?id=1MTntYoyzv_Y2veMiC90eQqwF8m7GYJmM"
response = requests.get(url)
with open("model.tflite", 'wb') as f:
    f.write(response.content)
model = tf.lite.Interpreter(model_path="model.tflite")
model.allocate_tensors()

url1 = "https://drive.google.com/uc?id=1ApURHKn-qiDps2lZclpRGxoZdkyDjTyr"
session = requests.Session()
file_id="1ApURHKn-qiDps2lZclpRGxoZdkyDjTyr"
response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)
    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)

# response = requests.get(url1, stream=True)
with open("model1.tflite", 'wb') as f:
    f.write(response.content)
model1 = tf.lite.Interpreter(model_path="model1.tflite")
model1.allocate_tensors()

st.image("cropedLogo.png")
st.title("I-Care")
st.info("Medical Imaging Scan - Easy Healthcare for Anyone Anytime")
# message = st.chat_input("Say something")
# img = st.image_input()
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # with open((uploaded_file.name), "wb") as f:
    #   f.write(uploaded_file.getbuffer())
    # To read image file buffer with PIL
    image = Image.open(uploaded_file)
    image = image.resize((224, 224))
    # img = load_img(f"{uploaded_file.name}", target_size=(224,224))
    # Convert image to numpy array if needed
    # Display the uploaded image
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    # Save the uploaded file
    # save_uploaded_file(uploaded_file)
    img = rescale(image)
    output = predict(model,img).argmax()
    st.write(calories[output])
# if img is None:
#     pass
# else:
#     output = predict(model,rescale(img)).argmax()
#     st.write(output)
