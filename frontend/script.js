const predictBtn = document.getElementById("predictBtn");
const reviewInput = document.getElementById("review");
const result = document.getElementById("result");

predictBtn.addEventListener("click", async () => {
    const review = reviewInput.value.trim();

    if (!review) {
        result.textContent = "Please enter a review.";
        return;
    }

    result.textContent = "Analyzing...";

    try {
        const response = await fetch("http://127.0.0.1:8000/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                review: review
            })
        });

        if (!response.ok) {
            throw new Error("Prediction request failed");
        }

        const data = await response.json();

        result.textContent = `Sentiment: ${data.sentiment}`;

    } catch (error) {
        result.textContent = "Could not connect to the API.";
        console.error(error);
    }
});