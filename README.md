# Ai-or-not---Detect-AI-generated-images
Detect if images are AI generated or not
 
Link to the web app: **[AI or Not - Web App](https://huggingface.co/spaces/Ohajoui/AI-or-Not)**
 
 ## Goal:
In view of the recent advancements of AI, and specifically AI generated images, I thought it could be useful to have an app with a trained model to detect if the images provided are AI generated or authentic (Not AI generated).
Thankfully as HuggingFace was hosting a competition on the topic and providing a data set to develop the model, I seized the chance to develop the shared app.
 
## Steps:
In order to develop the app, I proceeded through the following:
* Collected the data provided by **[HuggingFace](https://huggingface.co/datasets/competitions/aiornot)**
* Split the Training set to training & Validation set, and kept a Testing set untouched
* Leveraged Pytorch librairy to conduct the training/testing development of the model
  *  *Note: I did start with a pre-trained model: EfficientNet-b4*
  *  The choice for the EfficientNet-b4 was done mainly based on the high accuracy, robustness, efficiency,  and transfer learning capabilities of the model
  *  The previous was a conclusion made following this **[paper](https://arxiv.org/pdf/1905.11946.pdf "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks
")**  - I fully acknowledge many other models/options could be tested in our case *(something to explore in the future)*
* Deployed the model as web-app using Gradio for the front-end and hosting it through the HuggingFace Spaces option
* The app also has some examples available for the user to either test using those, or even providing their own images
 
## End-result
The web-app looks like the following:

![01](https://github.com/Oussamahajoui/Ai-or-not---Detect-AI-generated-images/assets/83676274/6db39bb4-af90-4ed1-80be-8df19e8f0507)

![02](https://github.com/Oussamahajoui/Ai-or-not---Detect-AI-generated-images/assets/83676274/7ab244b9-51f4-4050-98d0-f357cace47a5)

![03](https://github.com/Oussamahajoui/Ai-or-not---Detect-AI-generated-images/assets/83676274/7a620987-df11-484d-a0c6-24aa1098e9e8)

![04](https://github.com/Oussamahajoui/Ai-or-not---Detect-AI-generated-images/assets/83676274/2bafd0fe-1900-4b95-b465-80a0ef87864a)

![05](https://github.com/Oussamahajoui/Ai-or-not---Detect-AI-generated-images/assets/83676274/1efdda6c-c4c2-4687-b2f2-5916c23c4522)

## Next steps:
As next steps for this web-app:
* Need to add a flagging system to be able to catch user-feedback in the cases of wrong guesses by the model
* Other models should also be trained and tested on this data set to find the best alternatives - As many Machine Learning professionals say it's an iterative process and we need to try many options
* I would like to leverage tools like Weights & Biases, Cloud providers, Docker, etc. to be able to fully cover the total extent of Machine Learning app development as done in industry.
 
###### Made by Oussama Hajoui
