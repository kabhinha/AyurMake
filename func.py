from keras.models import load_model
from PIL import Image, ImageOps 
import numpy as np 
from langchain.llms import OpenAI
import os

apiKey = ""
os.environ["OPENAI_API_KEY"] = apiKey
llm = OpenAI(temperature=0.3)


def desc(plt):
    prompt = f'you are a Botanists and tell me benefits of "{plt}" in "200 words"'
    return llm.predict(prompt)


def process(ImagePath, model, labels)->dict:
    '''
    takes the following parameters and predicts the image among alovera, neem, tulsi, nagfani
    ImagePath: path of image to scan
    model: path of model
    labels: path of labels
    returns a dictionary
    keys as:
        class: predict which class(leaf)
        confidence: how much it is sure about the prediction
    '''
    np.set_printoptions(suppress=True)
    model = load_model(model, compile=False)
    class_names = open(labels, "r").readlines()
    data = np.ndarray(shape=(1, 150, 150, 3), dtype=np.float32)
    image = Image.open(ImagePath).convert("RGB")
    size = (150, 150)
    image = ImageOps.fit(image, size, Image.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    return {"class": class_name, "confidence": confidence_score}


if __name__=="__main__":
    print(process(r"E:\Users\HP\Desktop\aalix clg\projects\climate leaves\ayurmake\imagedataset\test\Crape_Jasmine\TD-S-037.jpg", r"E:\Users\HP\Desktop\aalix clg\projects\climate leaves\ayurmake\model\model_sev.h5", r"E:\Users\HP\Desktop\aalix clg\projects\climate leaves\ayurmake\model\labels.txt"))