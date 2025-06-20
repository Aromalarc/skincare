import requests

url = 'http://127.0.0.1:5000/predict'
image_path = 'C:\Users\aroma\Pictures\skin_images' 
with open(image_path, 'rb') as img:
    files = {'file': img}
    response = requests.post(url, files=files)

print("Status code:", response.status_code)
print("Response JSON:", response.json())
