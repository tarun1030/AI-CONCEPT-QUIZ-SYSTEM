from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from sentence_transformers import SentenceTransformer, util
import logging
import asyncio
import random
from pathlib import Path

app = FastAPI()

# Static and templates setup
app.mount("/static", StaticFiles(directory="server/static"), name="static")
templates = Jinja2Templates(directory="server/Templates")

# Model setup
model = SentenceTransformer("all-MiniLM-L6-v2")

# Logging setup
log_file = Path("logs/server.log")
log_file.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(filename=log_file, level=logging.INFO, format="%(asctime)s - %(message)s")

# Questions and answers
questions = [
    {"question": "What is Artificial Intelligence?", 
     "answer": "Artificial intelligence (AI) refers to the simulation of human intelligence in machines. AI systems are designed to mimic human cognitive functions, such as learning, reasoning, problem-solving, perception, and language understanding. AI technologies can be used for a wide range of tasks, from automating routine processes to making decisions based on complex data. AI is broadly classified into narrow AI, which is designed for specific tasks, and general AI, which aims to perform any intellectual task that a human can do."},
     
    {"question": "What is Machine Learning?", 
     "answer": "Machine learning (ML) is a subset of artificial intelligence that enables machines to learn from data and improve their performance without being explicitly programmed. ML algorithms use statistical techniques to identify patterns and relationships in large datasets. The more data a machine learns from, the better it can make predictions or decisions based on that data. There are various types of machine learning: supervised learning, unsupervised learning, and reinforcement learning, each with its own applications."},
     
    {"question": "What is Deep Learning?", 
     "answer": "Deep learning is a subset of machine learning that uses artificial neural networks to model and solve complex problems. These neural networks are designed to simulate the way the human brain processes information, with multiple layers of nodes (neurons) that can learn to represent various features of data. Deep learning is particularly powerful for tasks such as image and speech recognition, natural language processing, and autonomous systems. It requires large amounts of data and computational power to train the models, but it often yields highly accurate results."},
     
    {"question": "What is Supervised Learning?", 
     "answer": "Supervised learning is a type of machine learning in which the model is trained on a labeled dataset, where both the input data and the corresponding correct output (label) are provided. The algorithm learns to map the input to the correct output by identifying patterns in the data. This type of learning is commonly used for classification and regression tasks. For example, predicting whether an email is spam or not (classification) or estimating house prices based on various features (regression)."},
     
    {"question": "What is Unsupervised Learning?", 
     "answer": "Unsupervised learning is a machine learning technique that involves training a model on unlabeled data, where the algorithm tries to identify hidden patterns or structures in the data. Unlike supervised learning, there are no predefined labels or outcomes. Common techniques used in unsupervised learning include clustering (grouping similar data points) and dimensionality reduction (reducing the number of features in the dataset while retaining important information). Unsupervised learning is useful for tasks like customer segmentation and anomaly detection."},
     
    {"question": "What is Reinforcement Learning?", 
     "answer": "Reinforcement learning (RL) is a type of machine learning where an agent learns to make decisions by interacting with its environment. The agent receives feedback in the form of rewards or penalties based on its actions, and the goal is to maximize the cumulative reward over time. RL is used in applications where decision-making is sequential, such as robotics, gaming (e.g., AlphaGo), and autonomous vehicles. The agent uses trial and error to discover the best strategies for achieving its goals."},
     
    {"question": "What is Natural Language Processing?", 
     "answer": "Natural language processing (NLP) is a branch of artificial intelligence focused on the interaction between computers and human languages. The goal of NLP is to enable machines to understand, interpret, and generate human language in a way that is both meaningful and useful. NLP encompasses various tasks, including text classification, sentiment analysis, machine translation, and chatbot development. NLP plays a key role in applications like virtual assistants (e.g., Siri, Alexa) and language translation services."},
     
    {"question": "What is a Neural Network?", 
     "answer": "A neural network is a computational model inspired by the structure and functioning of the human brain. It consists of layers of interconnected nodes (neurons), where each node processes input data and passes it to the next layer. Neural networks are used for pattern recognition tasks, such as image classification, speech recognition, and natural language understanding. The network 'learns' by adjusting the weights of connections between neurons based on the data and the desired output. Deep neural networks, which contain many layers, are used in deep learning applications."},
     
    {"question": "What is Computer Vision?", 
     "answer": "Computer vision is a field of artificial intelligence that enables computers to interpret and understand visual information from the world, such as images and videos. Using algorithms and machine learning techniques, computer vision systems can recognize objects, detect motion, perform facial recognition, and more. The technology is used in various industries, including healthcare (e.g., medical imaging), automotive (e.g., self-driving cars), and entertainment (e.g., augmented reality). The goal of computer vision is to make sense of visual data in the same way humans do."},
     
    {"question": "What is Data Preprocessing?", 
     "answer": "Data preprocessing is a critical step in the machine learning pipeline that involves cleaning, transforming, and organizing raw data before it is used to train a model. Preprocessing helps ensure that the data is consistent, accurate, and in a format suitable for analysis. Common tasks include handling missing values, normalizing or standardizing numerical data, encoding categorical variables, and removing duplicates. Effective data preprocessing improves the performance and accuracy of machine learning models."},
     
    {"question": "What is Overfitting?", 
     "answer": "Overfitting occurs when a machine learning model becomes too complex and learns the details and noise in the training data to the extent that it negatively impacts the model's performance on new, unseen data. This happens when the model fits the training data too closely, capturing patterns that do not generalize well. Overfitting can be mitigated through techniques such as cross-validation, regularization, and pruning decision trees to ensure that the model remains robust and generalizes well to new data."},
     
    {"question": "What is Underfitting?", 
     "answer": "Underfitting occurs when a machine learning model is too simple to capture the underlying patterns in the data. It results in poor performance on both the training data and new, unseen data. Underfitting typically happens when the model has too few parameters or is overly constrained. To address underfitting, you can increase the complexity of the model, add more features, or use more powerful algorithms to capture the patterns in the data."},
     
    {"question": "What is Gradient Descent?", 
     "answer": "Gradient descent is an optimization algorithm used to minimize the loss function of a machine learning model. It works by iteratively adjusting the model's parameters in the direction of the steepest decrease in the loss, which is determined by the gradient of the function. The goal is to find the optimal set of parameters that minimizes the error between predicted and actual values. Variants of gradient descent, such as stochastic gradient descent (SGD) and mini-batch gradient descent, are commonly used to speed up the process and handle large datasets."},
     
    {"question": "What is a Decision Tree?", 
     "answer": "A decision tree is a supervised machine learning algorithm used for classification and regression tasks. It works by recursively splitting the dataset into subsets based on the values of input features, resulting in a tree-like structure of decisions. Each internal node of the tree represents a decision based on a feature, while each leaf node represents an outcome or prediction. Decision trees are easy to interpret and visualize but can suffer from overfitting if not properly pruned."},
     
    {"question": "What is Clustering?", 
     "answer": "Clustering is an unsupervised machine learning technique used to group similar data points together based on certain features. The goal is to identify natural clusters or groups in the data, with similar items placed in the same group. Clustering is widely used in exploratory data analysis, customer segmentation, and anomaly detection. Popular clustering algorithms include k-means, hierarchical clustering, and DBSCAN."},
     
    {"question": "What is Transfer Learning?", 
     "answer": "Transfer learning is a technique in machine learning where a pre-trained model is used on a new task. The idea is to leverage knowledge gained from a related task to save time and computational resources when training a new model. Transfer learning is commonly used in deep learning, where large neural networks pre-trained on massive datasets (like ImageNet) can be fine-tuned to perform well on smaller, task-specific datasets. This approach has made it easier to apply deep learning to domains with limited data."},
     
    {"question": "What is Bias in Machine Learning?", 
     "answer": "Bias in machine learning refers to systematic errors introduced by incorrect assumptions or simplifications in the learning algorithm. Bias can result from the choice of the model, the data used to train it, or the features selected for learning. High bias can lead to underfitting, where the model is too simplistic and unable to capture the underlying patterns in the data. Minimizing bias is crucial for building accurate and generalizable models."},
     
    {"question": "What is Variance in Machine Learning?", 
     "answer": "Variance refers to the model's sensitivity to fluctuations in the training data. A high-variance model learns the details and noise of the training data too well, which can lead to overfitting. On the other hand, a model with low variance may be too rigid and fail to capture important patterns in the data. Balancing bias and variance is key to building models that generalize well to new data."},
     
    {"question": "What is the Turing Test?", 
     "answer": "The Turing Test, proposed by the British mathematician and computer scientist Alan Turing in 1950, is a method of determining whether a machine can exhibit intelligent behavior equivalent to or indistinguishable from that of a human. In the test, a human evaluator interacts with both a machine and a human, without knowing which is which, and determines if the machine can successfully mimic human responses. The Turing Test has been a foundational concept in AI and remains a benchmark for evaluating machine intelligence."},
     
    {"question": "What is Big Data?", 
     "answer": "Big data refers to datasets that are so large and complex that traditional data processing methods are insufficient to handle them. These datasets typically come from diverse sources, such as social media, sensors, transactions, and IoT devices. Big data can be characterized by the three V's: volume (the amount of data), velocity (the speed at which data is generated and processed), and variety (the different types and formats of data). Big data technologies, such as Hadoop and Spark, are used to store, process, and analyze these massive datasets to extract valuable insights."},
     
    {"question": "What is Feature Engineering?", 
     "answer": "Feature engineering is the process of selecting, modifying, or creating new features from raw data to improve the performance of machine learning models. This step is crucial for making sure the model has access to the most relevant information. Feature engineering can include tasks like encoding categorical variables, scaling numerical features, handling missing data, and creating new features through mathematical transformations or domain-specific knowledge. Good feature engineering can significantly improve model accuracy."},
     
    {"question": "What is an Activation Function?", 
     "answer": "An activation function is a mathematical function that determines the output of a neural network node, or neuron, based on its input. The function introduces non-linearity into the network, allowing it to model complex patterns in the data. Common activation functions include the sigmoid function, ReLU (Rectified Linear Unit), and tanh function. The choice of activation function affects the network's ability to learn and generalize."},
     
    {"question": "What is Hyperparameter Tuning?", 
     "answer": "Hyperparameter tuning involves selecting the best set of hyperparameters for a machine learning model to optimize its performance. Hyperparameters are parameters that are not learned during training, such as the learning rate, number of layers, or the regularization strength. The tuning process typically involves experimenting with different values for these hyperparameters and evaluating the model's performance on validation data. Methods like grid search and random search are often used for hyperparameter optimization."},
     
    {"question": "What is a Support Vector Machine (SVM)?", 
     "answer": "A support vector machine (SVM) is a supervised machine learning algorithm commonly used for classification tasks. SVM works by finding the hyperplane that best separates the data into different classes. The algorithm aims to maximize the margin between the classes, ensuring that the boundary between them is as wide as possible. SVM can also be adapted for regression tasks and is particularly effective in high-dimensional spaces."},
     
    {"question": "What is the difference between AI, ML, and DL?", 
     "answer": "AI (Artificial Intelligence) is the broader field focused on creating machines that can simulate human intelligence, including tasks like learning, reasoning, and decision-making. ML (Machine Learning) is a subset of AI that focuses on building algorithms that enable machines to learn from data and improve over time. DL (Deep Learning) is a further subset of ML, which involves training artificial neural networks with many layers to solve complex tasks such as image recognition, natural language processing, and speech recognition."}
]


from fastapi import WebSocket, WebSocketDisconnect
import random
import logging

connected_clients = set()

@app.get("/", response_class=HTMLResponse)
async def home():
    return templates.TemplateResponse("index.html", {"request": {}})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.add(websocket)
    user_score = 0
    question_index = 0

    try:
        await websocket.send_text("Welcome to the AI Quiz! Answer the following questions.")
        random.shuffle(questions)

        while True:
            if question_index >= len(questions):
                # Final score message
                await websocket.send_text(f"Quiz ended. Your final score: {user_score}.")
                break

            current_question = questions[question_index]
            question, correct_answer = current_question["question"], current_question["answer"]
            question_index += 1

            await websocket.send_text(f"Question {question_index}: {question}")
            response = await websocket.receive_text()

            # Check if the user wants to end the quiz
            if response.lower() == "no":
                await websocket.send_text(f"Quiz ended. Your final score: {user_score}.")
                break

            # Calculate similarity score
            similarity = util.cos_sim(
                model.encode(response, convert_to_tensor=True),
                model.encode(correct_answer, convert_to_tensor=True)
            ).item()

            # Determine feedback and update score
            if similarity > 0.98:
                feedback = "Excellent! You nailed it! 😁"
                user_score += 5
            elif 0.95 < similarity <= 0.98:
                feedback = "Great! You're very close. 🙂"
                user_score += 4
            elif 0.90 < similarity <= 0.95:
                feedback = "Good effort! Almost there. 🤨"
                user_score += 3
            elif 0.85 < similarity <= 0.90:
                feedback = "Keep trying! You can do it! 😐"
                user_score += 2
            elif 0.80 < similarity <= 0.85:
                feedback = "Not quite right. Review and try again. 😓"
                user_score += 1
            else:
                feedback = "Incorrect. Don't give up! ❌"

            # Send feedback and current score
            await websocket.send_text(feedback)
            await websocket.send_text(f"Your current score: {user_score}/{question_index * 5}")

            # Wait for the user to decide to continue or end the quiz
            move_to_next = await websocket.receive_text()
            if move_to_next.lower() == "yes":
                continue
            elif move_to_next.lower() == "no":
                await websocket.send_text(f"Quiz ended. Your final score: {user_score}.")
                break
            else:
                await websocket.send_text("Invalid input. Please reply with 'yes' to continue or 'no' to end the quiz.")

        # Final message
        await websocket.send_text("Thank you for joining the AI Quiz! 🎉 Have a great day!")



    except WebSocketDisconnect:
        logging.info("Client disconnected")
        connected_clients.remove(websocket)
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        await websocket.send_text("An error occurred. Please try again later.")