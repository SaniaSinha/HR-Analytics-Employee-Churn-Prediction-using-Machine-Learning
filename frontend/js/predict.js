
const API_URL = "http://127.0.0.1:8000";

const form = document.getElementById("predictionForm");
const resultDiv = document.getElementById("predictionResult");

form.addEventListener("submit", async function (event) {
    event.preventDefault();

    const formData = new FormData(form);
    const data = Object.fromEntries(formData.entries());

    // Convert numeric inputs into numbers
    data.satisfaction_level = Number(data.satisfaction_level);
    data.last_evaluation = Number(data.last_evaluation);
    data.number_project = Number(data.number_project);
    data.average_monthly_hours = Number(data.average_monthly_hours);
    data.time_spend_company = Number(data.time_spend_company);
    data.Work_accident = Number(data.Work_accident);
    data.promotion_last_5years = Number(data.promotion_last_5years);

    resultDiv.innerHTML = `
        <div class="alert alert-info">
            Selecting the best model and running prediction...
            Please wait.
        </div>
    `;

    const button = form.querySelector('button[type="submit"]');
    button.disabled = true;
    button.textContent = "Predicting...";

    try {
        const response = await fetch(`${API_URL}/predict`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(data)
        });

        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.detail || "Prediction failed.");
        }

        const prediction = result.prediction;
        const probability = Number(result.probability);
        const f1Score = Number(result.f1_score);

        const isLeaving = prediction === 1 || prediction === "1";

        resultDiv.innerHTML = `
            <div class="alert ${isLeaving ? "alert-danger" : "alert-success"} p-4">

                <h4 class="fw-bold mb-3">
                    ${isLeaving
                        ? "Employee Likely to Leave"
                        : "Employee Likely to Stay"}
                </h4>

                <p class="mb-2">
                    Prediction:
                    <strong>
                        ${isLeaving ? "Will Leave (1)" : "Will Stay (0)"}
                    </strong>
                </p>

                <p class="mb-2">
                    Automatically Selected Experiment:
                    <strong>${result.experiment}</strong>
                </p>

                <p class="mb-2">
                    Selected Model F1-score:
                    <strong>
                        ${Number.isFinite(f1Score)
                            ? (f1Score * 100).toFixed(2) + "%"
                            : "Not available"}
                    </strong>
                </p>

                <p class="mb-0">
                    Churn Probability:
                    <strong>
                        ${Number.isFinite(probability)
                            ? (probability * 100).toFixed(2) + "%"
                            : "Not available"}
                    </strong>
                </p>

            </div>
        `;

    } catch (error) {
        resultDiv.innerHTML = `
            <div class="alert alert-danger">
                <strong>Prediction Error:</strong> ${error.message}
                <br><br>
                Please check that your FastAPI backend is running.
            </div>
        `;
    } finally {
        button.disabled = false;
        button.textContent = "Predict Employee Churn";
    }
});
