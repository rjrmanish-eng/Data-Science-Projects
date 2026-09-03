import requests

url = "https://www.fast2sms.com/dev/bulkV2"

payload = {
    'route': 'q',
    'message': 'Hello from Python!',
    'language': 'english',
    'numbers': '8209226015'
}

headers = {
    'authorization': 'uO2Yp1MQlvwgCh8rGTnbsyHFoD70dZAW9VXmfL6PejkiacK3x4M9hPHZIyTKsv2ojSDkLt7cdrFRzBim'
}

response = requests.post(url, data=payload, headers=headers)

print(response.text)