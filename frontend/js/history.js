
const API_URL = "http://127.0.0.1:8000";

async function loadHistory() {
    const loading = document.getElementById("historyLoading");
    const error = document.getElementById("historyError");
    const content = document.getElementById("historyContent");
    const warning = document.getElementById("databaseWarning");
    const tableBody = document.getElementById("historyTableBody");
    const emptyHistory = document.getElementById("emptyHistory");

    loading.classList.remove("d-none");
    error.classList.add("d-none");
    content.classList.add("d-none");
    warning.classList.add("d-none");

    try {
        const response = await fetch(`${API_URL}/history`);

        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.detail || "Unable to load history.");
        }

        const history = Array.isArray(result)
            ? result
            : result.history || result.predictions || result.data || [];

        const databaseConnected =
            result.database_connected !== false &&
            result.mongo !== "disconnected";

        if (!databaseConnected) {
            warning.classList.remove("d-none");
        }

        tableBody.innerHTML = "";

        if (history.length === 0) {
            emptyHistory.classList.remove("d-none");
            document.querySelector(".table-responsive").classList.add("d-none");
        } else {
            emptyHistory.classList.add("d-none");
            document.querySelector(".table-responsive").classList.remove("d-none");

            history.forEach((item, index) => {
                const prediction = Number(item.prediction);
                const isLeaving = prediction === 1;

                const probability = Number(
                    item.probability ?? item.churn_probability
                );

                const f1Score = Number(item.f1_score);

                const timestamp =
                    item.timestamp || item.created_at || item.date;

                const date = timestamp
                    ? new Date(timestamp).toLocaleString()
                    : "Not available";

                const row = document.createElement("tr");

                const values = [
                    index + 1,
                    date,
                    isLeaving ? "Will Leave" : "Will Stay",
                    Number.isFinite(probability)
                        ? (probability * 100).toFixed(2) + "%"
                        : "Not available",
                    item.experiment || "Not available",
                    Number.isFinite(f1Score)
                        ? (f1Score * 100).toFixed(2) + "%"
                        : "Not available"
                ];

                values.forEach((value, cellIndex) => {
                    const cell = document.createElement("td");
                    cell.textContent = value;

                    if (cellIndex === 2) {
                        cell.className = isLeaving
                            ? "status-leave"
                            : "status-stay";
                    }

                    row.appendChild(cell);
                });

                tableBody.appendChild(row);
            });
        }

        content.classList.remove("d-none");

    } catch (err) {
        error.textContent = "Error: " + err.message;
        error.classList.remove("d-none");
    } finally {
        loading.classList.add("d-none");
    }
}

document.addEventListener("DOMContentLoaded", loadHistory);
