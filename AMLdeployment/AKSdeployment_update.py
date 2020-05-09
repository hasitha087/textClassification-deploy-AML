# You can update the same AKS service when you want to update next time(REST end points won't change)

from azureml.core.model import Model
from azureml.core import Workspace
from azureml.core.model import InferenceConfig
from azureml.core.webservice import Webservice

ws = Workspace.from_config()
service_name = 'sentimentclassifier'

model_1 = Model(ws, name='onboard_sentiment_vector.pkl')
model_2 = Model(ws, name='onboard_sentiment_classifier.pkl')
model_3 = Model(ws, name='onboard_vector.pkl')
model_4 = Model(ws, name='onboard_classifier.pkl')


inference_config = InferenceConfig(runtime="python",
                                   entry_script="AMLscore.py",
                                   conda_file="aml_config/myenv.yml")


# Retrieve existing service.
service = Webservice(name=service_name, workspace=ws)

service.update(models=[model_1, model_2, model_3, model_4], inference_config=inference_config)
print(service.state)
print(service.get_logs())