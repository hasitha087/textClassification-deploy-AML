from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core import Workspace
from azureml.core.webservice import AciWebservice
from azureml.core.model import Model, InferenceConfig
import json

# Get AML workspace details
ws = Workspace.from_config()
with open("aml_config/config.json") as f:
    config = json.load(f)

workspace_name = config["workspace_name"]
resource_group = config["resource_group"]
subscription_id = config["subscription_id"]
location = config["location"]
tenant_id = config["tenant_id"]
service_principal_id = config["service_principal_id"]
service_principal_password = config["service_principal_password"]

# Service principle authentication
svc_pr = ServicePrincipalAuthentication(
    tenant_id=tenant_id,
    service_principal_id=service_principal_id,
    service_principal_password=service_principal_password)

Workspace(
    subscription_id=subscription_id,
    resource_group=resource_group,
    workspace_name=workspace_name,
    auth=svc_pr
)

print("Successfully Authenticated. Found workspace {} at location {}".format(ws.name, ws.location))

imageName = "sentimentclassifier-aci"

# Register model on AML service
Model.register(model_path='../models/onboard_sentiment_vector.pkl', model_name='onboard_sentiment_vector.pkl',
                           workspace=ws)
Model.register(model_path='../models/onboard_sentiment_classifier.pkl', model_name='onboard_sentiment_classifier.pkl',
                            workspace=ws)
Model.register(model_path='../models/onboard_vector.pkl', model_name='onboard_vector.pkl',
                           workspace=ws)
Model.register(model_path='../models/onboard_classifier.pkl', model_name='onboard_classifier.pkl',
                            workspace=ws)

# Load models from model registery
model_1 = Model(ws, name='onboard_sentiment_vector.pkl')
model_2 = Model(ws, name='onboard_sentiment_classifier.pkl')
model_3 = Model(ws, name='onboard_vector.pkl')
model_4 = Model(ws, name='onboard_classifier.pkl')

# Create configurations
inference_config = InferenceConfig(runtime="python",
                                   entry_script="AMLscore.py",
                                   conda_file="aml_config/myenv.yml")

# Deploy on ACI
deployment_config = AciWebservice.deploy_configuration(cpu_cores = 1, memory_gb = 1)
service = Model.deploy(ws, imageName, [model_1, model_2, model_3, model_4], inference_config, deployment_config)
service.wait_for_deployment(show_output = True)
print(service.state)