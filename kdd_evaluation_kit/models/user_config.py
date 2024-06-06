# Importing DummyModel from the models package.
# The DummyModel class is located in the dummy_model.py file inside the 'models' directory.
from models.dummy_model import DummyModel, llama3_8b_FewShot, llama3_8b_FewShot_vllm, aya23_fewshot_VLLM, glm_9b_FewShot_vllm

# This line establishes an alias for the DummyModel class to be used within this script.
# Instead of directly using DummyModel everywhere in the code, we're assigning it to 'UserModel'.
# This approach allows for easier reference to your model class when evaluating your models,
UserModel = glm_9b_FewShot_vllm
# When implementing your own model please follow this pattern:
#
# from models.your_model import YourModel
#
# Replace 'your_model' with the name of your Python file containing the model class
# and 'YourModel' with the class name of your model.
#
# Finally, assign YourModel to UserModel as shown below to use it throughout your script.
#
# UserModel = YourModel

