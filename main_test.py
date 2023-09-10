import requests
import json

# URL to send the POST request to
url = 'http://127.0.0.1:5000'

# Open the file you want to send as 'image'
with open('image.jpeg', 'rb') as file:
    # Create a dictionary with the form data
    form_data = {'image': ('image.jpeg', file)}

    # Send the POST request with the form data
    response = requests.post(url, files=form_data)

# Check the response
if response.status_code == 200:
    print('POST request successful.')
    response_json = response.json()
    print(type(json.dumps(response_json)))
else:
    print(f'POST request failed with status code {response.status_code}')
    print(response.text)
