const websocket = new WebSocket("ws://localhost:8000/ws");
let quizEnded = false;  // Flag to track whether the quiz has ended
let finalScore = 0;  // Store the final score
let totalQuestions = 1;  // Dynamically track total questions (based on "Next" button clicks)
let currentScore = 0; // Store the current score

websocket.onopen = () => {
    document.getElementById("message").innerText = "Connected to server.";
};

websocket.onmessage = (event) => {
    const message = event.data;

    if (quizEnded) {
        return;  // If the quiz has ended, stop processing further messages
    }

    if (message.startsWith("Question")) {
        document.getElementById("question").innerText = message;
        document.getElementById("feedback").innerText = "";
        document.getElementById("comment").innerText = "";
        document.getElementById("response").value = "";
        document.getElementById("response").disabled = false;
        document.getElementById("response").style.visibility = "visible";
        document.getElementById("score").style.visibility = "hidden"; // Hide score during question display
        toggleButtons("submit");
        document.getElementById("message").innerText = "Please answer the question.";
    } else if (
        message.startsWith("Excellent") || 
        message.startsWith("Great") || 
        message.startsWith("Good effort") || 
        message.startsWith("Keep trying") || 
        message.startsWith("Incorrect")
    ) {
        document.getElementById("comment").innerText = message;
        document.getElementById("response").style.visibility = "hidden";
        
        // Update current score based on feedback
        if (message.startsWith("Excellent") || message.startsWith("Great") || message.startsWith("Good effort")) {
            currentScore++; // Increase score on correct answers
        }

        // Display current score/total questions
        document.getElementById("score").innerText = `Current score: ${currentScore}/${totalQuestions}`;
        document.getElementById("score").style.visibility = "visible"; // Show score after feedback
        
        toggleButtons("next");
    } else if (message.startsWith("Your current score:")) {
        document.getElementById("score").innerText = message;
        finalScore = parseInt(message.split(":")[1].trim());  // Update the final score
    } else if (message.startsWith("Quiz ended")) {
        document.getElementById("message").innerText = `${message} Thank you for visiting.`;
        quizEnded = true;  // Mark quiz as ended
        // Hide question, score, and other content
        document.getElementById("question").innerText = "";
        document.getElementById("score").innerText = `Final score: ${finalScore}/${totalQuestions*5}`;
        toggleButtons("restart");  // Show Restart button
    } else {
        document.getElementById("message").innerText = message;
    }
};

websocket.onclose = () => {
    document.getElementById("message").innerText = "Disconnected from server.";
};

// Button event listeners
document.getElementById("submit-btn").onclick = () => {
    const response = document.getElementById("response").value.trim();
    if (response) {
        websocket.send(response);
        document.getElementById("response").value = "";
        document.getElementById("response").disabled = true;
        toggleButtons("none");
    } else {
        document.getElementById("feedback").innerText = "Please enter a response.";
    }
};

document.getElementById("next-btn").onclick = () => {
    totalQuestions++; // Increment total questions count
    websocket.send("yes");
    document.getElementById("response").disabled = false;
    document.getElementById("response").style.visibility = "visible";
    document.getElementById("response").value = "";
    toggleButtons("submit");
    document.getElementById("feedback").innerText = "";
    document.getElementById("message").innerText = `Please answer question ${totalQuestions}.`;
};

document.getElementById("end-btn").onclick = () => {
    websocket.send("no");
    document.getElementById("response").disabled = true;
    document.getElementById("response").style.visibility = "hidden";
    document.getElementById("comment").innerText = "";
    quizEnded = true;  // Mark quiz as ended
    // Hide question, score, and feedback
    document.getElementById("question").innerText = "";
    document.getElementById("score").innerText = `Final score: ${finalScore}/${totalQuestions*5}`;
    toggleButtons("restart");  // Show Restart button
    document.getElementById("message").innerText = `Quiz ended. Thank you for visiting.`;
};

document.getElementById("restart-btn").onclick = () => {
    location.reload();  
};

// Utility function to toggle button visibility
function toggleButtons(visibleButton) {
    const submitBtn = document.getElementById("submit-btn");
    const nextBtn = document.getElementById("next-btn");
    const endBtn = document.getElementById("end-btn");
    const restartBtn = document.getElementById("restart-btn");

    submitBtn.style.display = visibleButton === "submit" ? "block" : "none";
    nextBtn.style.display = visibleButton === "next" ? "block" : "none";
    endBtn.style.display = visibleButton === "next" ? "block" : "none";
    restartBtn.style.display = visibleButton === "restart" ? "block" : "none";
}
