import requests

host = "molecule.eba-gtgfdcdy.us-east-1.elasticbeanstalk.com"
url = f'http://{host}/predict_app'

customer = {"molecule": "CC(C)C[C@H](NC(=O)[C@H](CC1:C:C:C:C:C:1)NC(=O)C1:C:N:C:C:N:1)B(O)O"}

response = requests.post(url, json=customer).json()
print(response)