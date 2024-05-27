# Simple chatbot for Vietnamese
## A. Introduction
**Chatbot (Large Language Models):** Chatbots are one of the most popular applications in the field of Artificial Intelligence. This is increasingly evident with the advent of LLMs in recent times, which are capable of performing numerous tasks with high accuracy and are still being developed and improved. Advances in this technology not only enhance the quality of interactions between humans and machines but also open up new possibilities in automation and personalized services.

It can simulate a conversation (or a chat) with a user in natural language through messaging applications, websites, mobile apps, or through telephone.

In this repo, I will build a chatbot using LLMs trained on the Vietnamese dataset [VinaLLaMA](https://huggingface.co/vilm/vinallama-7b).
## B. Technical overview
- Pipeline:
  Download and import libraries → Load trained model → Load dataset → Create training dataset → Training model → Running prediction.

Here are the detailed steps of implementation:
### Step 1: Install libraries

![image](https://github.com/nhphan/SimpleVNChatbot/assets/96032860/04325a86-9f5d-462e-8195-cd8c576f87b2)

### Step 2: Import libraries/modules

![image](https://github.com/nhphan/SimpleVNChatbot/assets/96032860/5f5a78a6-7ee9-484b-badf-709f6684f657)

### Step 3: Load trained model

![image](https://github.com/nhphan/SimpleVNChatbot/assets/96032860/068e66ed-4694-4529-a77a-3996e44978fe)

### Step 4: Configure LLMs

![image](https://github.com/nhphan/SimpleVNChatbot/assets/96032860/37c410bf-acd2-479b-9e75-43f5766d13a8)

### Step 5: Run trained model

![image](https://github.com/nhphan/SimpleVNChatbot/assets/96032860/9abf21f8-28a6-4396-9926-e659bd4cf8d9)

Following this sequence, we will be able to build a simple chatbot that can answer basic user questions. Here are some examples of input questions and the chatbot's responses:
![image](https://github.com/nhphan/SimpleVNChatbot/assets/96032860/6a00e505-b1dd-4a27-a2e8-c3df5a8b6e60)
![image](https://github.com/nhphan/SimpleVNChatbot/assets/96032860/ec554502-b843-4ef2-a3aa-587fc37ff743)

## C. How to install
Please refer to this video.
## D. Conclusion
In conclusion, we have developed a simple chatbot that can answer basic user questions for Vietnamese. The strengths of this chatbot include its ability to understand and respond to common queries, its ease of use, and its potential to improve with more data. However, there are also some weaknesses. The chatbot may not always understand complex questions. One way to improve the model's accuracy is through fine-tuning with a specific dataset to address a particular topic, such as mathematics or history. However, due to time constraints, this repository will only utilize a general dataset.
