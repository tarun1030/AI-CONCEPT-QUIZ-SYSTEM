import pytest
import websockets
import asyncio
from server.server import app  # Assuming your FastAPI app is named app
from fastapi.testclient import TestClient
from server.server import questions 
client = TestClient(app)

@pytest.mark.asyncio
async def test_websocket_quiz():
    # Test WebSocket connection and quiz functionality
    uri = "ws://localhost:8000/ws"  # WebSocket URI to connect to

    async with websockets.connect(uri) as websocket:
        # Check welcome message
        welcome_message = await websocket.recv()
        assert "Welcome to the AI Quiz!" in welcome_message

        user_score = 0
        question_index = 0
        total_questions = len(questions)  # Total number of questions

        while True:
            try:
                # Receive a question from the server
                question_message = await websocket.recv()
                assert "Question" in question_message  # Ensure we received a question

                # Send a response to the question (you may want to change this for each question)
                response = "Machine learning"  # Just an example, you can modify for different questions
                await websocket.send(response)

                # Receive feedback from the server
                feedback = await websocket.recv()
                assert any(feedback.startswith(text) for text in [
                    "Excellent!", "Great!", "Good effort!", "Keep trying!", "Not quite right.", "Incorrect."])

                # Receive current score
                score_message = await websocket.recv()
                assert "Your current score" in score_message

                # Move to the next question
                await websocket.send("yes")  # Simulating a response to move to next question
                question_index += 1

            except websockets.exceptions.ConnectionClosed:
                break

            # Check if all questions are completed and the quiz has ended
            if question_index >= total_questions:
                # We reached the last question, now check for final score
                final_score_message = await websocket.recv()
                assert "Quiz ended" in final_score_message
                assert "Your final score" in final_score_message
                print(final_score_message)
                break  # Exit loop after receiving final score

@pytest.mark.asyncio
async def test_home_page():
    # Test the home page HTML response
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

    # Check for the AI Quiz title (h1)
    assert "<h1>AI Quiz</h1>" in response.text

    # Check for the presence of the quiz container
    assert '<div id="quiz-container">' in response.text

    # Ensure that an input field is present
    assert '<input type="text" id="response"' in response.text

    # Check for the submit button
    assert '<button id="submit-btn"' in response.text