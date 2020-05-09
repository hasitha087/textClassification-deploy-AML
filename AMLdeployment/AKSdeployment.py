from azureml.core.webservice import AksWebservice
from azureml.core.model import Model
from azureml.core import Workspace
from azureml.core.compute import AksCompute
from azureml.core.model import InferenceConfig

ws = Workspace.from_config()
imageName = "sentimentclassifier"

# Load models from model registry
model_1 = Model(ws, name='onboard_sentiment_vector.pkl')
model_2 = Model(ws, name='onboard_sentiment_classifier.pkl')
model_3 = Model(ws, name='onboard_vector.pkl')
model_4 = Model(ws, name='onboard_classifier.pkl')


inference_config = InferenceConfig(runtime="python",
                                   entry_script="AMLscore.py",
                                   conda_file="aml_config/myenv.yml")

# Deploy on AKS cluster
aks_target = AksCompute(ws,"clus-aks")
deployment_config = AksWebservice.deploy_configuration(cpu_cores = 1, memory_gb = 1, auth_enabled=False)  #auth_enabled=False, token_auth_enabled=True
service = Model.deploy(ws, imageName, [model_1, model_2, model_3, model_4], inference_config, deployment_config, aks_target)
service.wait_for_deployment(show_output = True)
print(service.state)
print(service.get_logs())