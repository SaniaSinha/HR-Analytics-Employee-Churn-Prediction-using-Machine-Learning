
const API_URL = "http://127.0.0.1:8000";

async function loadDashboard() {
    const errorDiv = document.getElementById("dashboardError");
    const warningDiv = document.getElementById("databaseWarning");
    const recentTable = document.getElementById("recentPredictions");

    try {
        const response = await fetch(`${API_URL}/dashboard`);
        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.detail || "Unable to load dashboard.");
        }

        document.getElementById("totalPredictions").textContent =
            result.total_predictions ?? 0;

        document.getElementById("employeesLeft").textContent =
            result.employees_left ?? 0;

        document.getElementById("employeesStayed").textContent =
            result.employees_stayed ?? 0;

        if (
            result.database_connected === false ||
            result.mongo === "disconnected"
        ) {
            warningDiv.classList.remove("d-none");
        } else {
            warningDiv.classList.add("d-none");
        }

        const predictions =
            result.recent_predictions || result.recent || [];

        recentTable.innerHTML = "";

        if (predictions.length === 0) {
            recentTable.innerHTML = `
                <tr>
                    <td colspan="3" class="text-center text-secondary">
                        No prediction records available yet.
                    </td>
                </tr>
            `;
            return;
        }

        predictions.forEach(item => {
            const isLeaving = Number(item.prediction) === 1;

            const probability = Number(
                item.probability ?? item.churn_probability
            );

            const timestamp =
                item.timestamp || item.created_at || item.date;

            const date = timestamp
                ? new Date(timestamp).toLocaleString()
                : "Not available";

            const row = document.createElement("tr");

            const dateCell = document.createElement("td");
            dateCell.textContent = date;

            const predictionCell = document.createElement("td");
            predictionCell.textContent = isLeaving
                ? "Will Leave"
                : "Will Stay";

            predictionCell.style.color = isLeaving
                ? "#ff7185"
                : "#20c997";

            const probabilityCell = document.createElement("td");
            probabilityCell.textContent = Number.isFinite(probability)
                ? (probability * 100).toFixed(2) + "%"
                : "Not available";

            row.append(dateCell, predictionCell, probabilityCell);
            recentTable.appendChild(row);
        });

    } catch (error) {
        errorDiv.textContent = "Dashboard Error: " + error.message;
        errorDiv.classList.remove("d-none");

        recentTable.innerHTML = `
            <tr>
                <td colspan="3" class="text-center text-danger">
                    Unable to load prediction records.
                </td>
            </tr>
        `;
    }
}

document.addEventListener("DOMContentLoaded", loadDashboard);
