from azureml.core.conda_dependencies import CondaDependencies

myenv = CondaDependencies()

myenv.add_pip_package("numpy")
myenv.add_pip_package("sklearn")
# myenv.add_conda_package("nltk")


with open("aml_config/myenv.yml", "w") as f:
    f.write(myenv.serialize_to_string())