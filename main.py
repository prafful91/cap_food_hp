from fastapi import FastAPI, File, UploadFile
import shutil,cv2
from ultralytics import YOLO
import os,pickle
from io import BytesIO
import numpy as np
from datetime import datetime
from PIL import Image
from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import tensorflow as tf
from tensorflow.keras.preprocessing import image as img_tf
from tensorflow.keras.models import load_model



import base64


app = FastAPI()

# Mount a directory containing static files (e.g., images)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up Jinja2Templates
templates = Jinja2Templates(directory="templates")

#Model Path

model_path = 'models/best.pt'






@app.post("/prediction_object/")
async def predict_images(request: Request, image1: UploadFile = File(...)):

    try:

        start_time = datetime.now()

        # Save uploaded images to the 'static' directory
        image1_path = os.path.join("temp_images", image1.filename)


        with open(image1_path, "wb") as f1:
            shutil.copyfileobj(image1.file, f1)
            img = image1.file

        for image in os.listdir('temp_images'):
            if not '.txt' in str(image):
                uploaded_image = os.path.join('temp_images',image)

        # Load a pretrained YOLOv8n model

        model = YOLO(model_path)
        results = model(uploaded_image)

        # Visualize the results on the frame
        obj_image = results[0].plot()
        image_np = np.array(obj_image)
        image_base64 = base64.b64encode(image_np.tobytes()).decode('utf-8')
        pred_image_path = 'temp_images/predict.png'
        pil_image = Image.fromarray(cv2.cvtColor(obj_image, cv2.COLOR_BGR2RGB))
        pil_image.save(pred_image_path)

        with open(pred_image_path,'rb') as f1:
            image_base64 = base64.b64encode(f1.read()).decode("utf-8")


    
        for image_path in os.listdir('./temp_images'):

            if image_path.endswith(".txt"):
                continue
            del_path = './temp_images/' + image_path

            if os.path.exists(del_path):
                os.remove(del_path)

        time_diff = datetime.now() - start_time
        total_seconds = round(time_diff.total_seconds(),1)


        return templates.TemplateResponse(
            "image_template.html",
            {
              "request": request,
              "image1": f"data:image/jpeg;base64,{image_base64}", 
              'total_seconds':total_seconds
              }
        )
    except Exception as e:

        for image_path in os.listdir('./temp_images'):

            if image_path.endswith(".txt"):
                continue
            del_path = './temp_images/' + image_path

            if os.path.exists(del_path):
                os.remove(del_path)
        print(f'This is the error---------{e}')        
        return templates.TemplateResponse(
            "image_template.html",
            {
              "request": request,
              
              }
            )

@app.get('/')
async def read_item(request:Request):
    
    return templates.TemplateResponse(
            "home.html",
            {"request": request}
        )


@app.get('/predict_object')
async def read_item(request:Request):
    
    return templates.TemplateResponse(
            "form_template.html",
            {"request": request}
        )


@app.get("/Interim_report")
async def open_pdf():
    pdf_path = "Reports\Capstone_Interim_Report.pdf"  # Replace with the actual path to your PDF file
    return FileResponse(pdf_path, media_type="application/pdf", headers={"Content-Disposition": "inline"})


class_model = load_model('models/Food_mobnet.h5')

@app.post("/prediction_classification/")
async def predict_images(request: Request, image1: UploadFile = File(...)):

    try:

        start_time = datetime.now()

        # Save uploaded images to the 'static' directory
        image1_path = os.path.join("temp_images", image1.filename)


        with open(image1_path, "wb") as f1:
            shutil.copyfileobj(image1.file, f1)
            img = image1.file

        for image in os.listdir('temp_images'):
            if not '.txt' in str(image):
                uploaded_image = os.path.join('temp_images',image)

        # Load a pretrained tensorflow model

        

        img_path = uploaded_image  # Replace with the path to your image
        img = img_tf.load_img(img_path, target_size=(224, 224))
        img_array = img_tf.img_to_array(img)
        img_array = img_array/255
        img_array = np.expand_dims(img_array, axis=0)

        # Visualize the results on the frame
        predictions = class_model.predict(img_array)
        print(predictions)
        predicted_class_index = int(predictions.argmax(axis=-1))
        with open('models/food_mobnet_class.pkl', 'rb') as file:
            class_labels = list(pickle.load(file))
        predicted_class_label = class_labels[predicted_class_index]
        

        with open(uploaded_image,'rb') as f1:
            image_base64 = base64.b64encode(f1.read()).decode("utf-8")


    
        for image_path in os.listdir('./temp_images'):

            if image_path.endswith(".txt"):
                continue
            del_path = './temp_images/' + image_path

            if os.path.exists(del_path):
                os.remove(del_path)

        time_diff = datetime.now() - start_time
        total_seconds = round(time_diff.total_seconds(),1)


        return templates.TemplateResponse(
            "image_template_classification.html",
            {
              "request": request,
              "image1": f"data:image/jpeg;base64,{image_base64}", 
              'total_seconds':total_seconds,
              'predicted_class':predicted_class_label
              }
        )
    except Exception as e:

        for image_path in os.listdir('./temp_images'):

            if image_path.endswith(".txt"):
                continue
            del_path = './temp_images/' + image_path

            if os.path.exists(del_path):
                os.remove(del_path)
        print(f'This is the error---------{e}')        
        return templates.TemplateResponse(
            "image_template_classification.html",
            {
              "request": request,
              
              }
            )


@app.get('/predict_classification')
async def read_item(request:Request):
    
    return templates.TemplateResponse(
            "form_template_classification.html",
            {"request": request}
        )