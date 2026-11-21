
const API_URL = "http://127.0.0.1:8000";

const samplingMethods = ["BASE", "SMOTE", "SMOTE-ENN"];

const modelNames = [
    "Logistic Regression",
    "KNN",
    "SVM",
    "Decision Tree",
    "Random Forest",
    "AdaBoost",
    "XGBoost"
];

let evaluationData = null;
let rocChart = null;


// =====================================================
// LOAD EVALUATION DATA
// =====================================================

async function loadEvaluation() {

    const loading = document.getElementById("loading");
    const error = document.getElementById("error");

    try {

        const response = await fetch(`${API_URL}/evaluation`);

        if (!response.ok) {
            throw new Error("Unable to load evaluation results.");
        }

        evaluationData = await response.json();

        loading.classList.add("d-none");

        document
            .getElementById("evaluationContent")
            .classList.remove("d-none");

        renderAllResults();

    } catch (err) {

        loading.classList.add("d-none");

        error.classList.remove("d-none");

        error.textContent =
            "Could not load evaluation results. Please ensure the FastAPI backend is running. " +
            err.message;
    }
}


// =====================================================
// RENDER ALL RESULTS
// =====================================================

function renderAllResults() {

    const results = evaluationData.results;

    document.getElementById("totalExperiments").textContent =
        results.length;

    renderComparisonTable("Accuracy", "accuracyTable");

    renderComparisonTable("Precision", "precisionTable");

    renderComparisonTable("Recall", "recallTable");

    renderComparisonTable("F1", "f1Table");

    renderComparisonTable("AUC", "aucTable");

    renderROCChart();

    renderConfusionMatrixSelector();
}


// =====================================================
// GENERATE COMPARISON TABLES
// =====================================================

function renderComparisonTable(metric, tableId) {

    const table = document.getElementById(tableId);

    let html = `
        <thead class="table-dark">
            <tr>
                <th>Sampling</th>
    `;

    modelNames.forEach(model => {
        html += `<th>${model}</th>`;
    });

    html += `</tr></thead><tbody>`;

    samplingMethods.forEach(sampling => {

        html += `<tr><th>${sampling}</th>`;

        modelNames.forEach(model => {

            const result = evaluationData.results.find(
                item =>
                    item.Sampling === sampling &&
                    item.Model === model
            );

            const value = result
                ? (result[metric] * 100).toFixed(2) + "%"
                : "N/A";

            html += `<td>${value}</td>`;
        });

        html += `</tr>`;
    });

    html += `</tbody>`;

    table.innerHTML = html;
}


// =====================================================
// ROC CURVE CHART
// BASE MODELS ONLY
// =====================================================

function renderROCChart() {

    const ctx = document
        .getElementById("rocChart")
        .getContext("2d");

    const datasets = [];

    modelNames.forEach(model => {

        const curve = evaluationData.roc_curves[model];

        if (!curve) return;

        datasets.push({
            label: `${model} (AUC: ${curve.auc.toFixed(3)})`,
            data: curve.fpr.map((x, index) => ({
                x: x,
                y: curve.tpr[index]
            })),
            showLine: true,
            fill: false,
            pointRadius: 0,
            borderWidth: 2
        });
    });

    datasets.push({
        label: "Random Classifier",
        data: [
            { x: 0, y: 0 },
            { x: 1, y: 1 }
        ],
        showLine: true,
        fill: false,
        pointRadius: 0,
        borderDash: [5, 5]
    });

    if (rocChart) {
        rocChart.destroy();
    }

    rocChart = new Chart(ctx, {
        type: "scatter",

        data: {
            datasets: datasets
        },

        options: {
            responsive: true,

            scales: {
                x: {
                    type: "linear",
                    min: 0,
                    max: 1,
                    title: {
                        display: true,
                        text: "False Positive Rate"
                    }
                },

                y: {
                    min: 0,
                    max: 1,
                    title: {
                        display: true,
                        text: "True Positive Rate"
                    }
                }
            },

            plugins: {
                legend: {
                    position: "bottom"
                }
            }
        }
    });
}


// =====================================================
// CONFUSION MATRIX SELECTOR
// =====================================================

function renderConfusionMatrixSelector() {

    const select = document.getElementById("confusionSelect");

    select.innerHTML = "";

    Object.keys(evaluationData.confusion_matrices).forEach(name => {

        const option = document.createElement("option");

        option.value = name;
        option.textContent = name;

        select.appendChild(option);
    });

    select.addEventListener("change", renderConfusionMatrix);

    renderConfusionMatrix();
}


// =====================================================
// DISPLAY SELECTED CONFUSION MATRIX
// =====================================================

function renderConfusionMatrix() {

    const selected = document.getElementById("confusionSelect").value;

    const matrix =
        evaluationData.confusion_matrices[selected];

    if (!matrix) return;

    const html = `
        <div class="table-responsive">
            <table class="table table-bordered text-center">
                <thead class="table-dark">
                    <tr>
                        <th></th>
                        <th>Predicted: Stay (0)</th>
                        <th>Predicted: Leave (1)</th>
                    </tr>
                </thead>

                <tbody>
                    <tr>
                        <th>Actual: Stay (0)</th>
                        <td>${matrix[0][0]}</td>
                        <td>${matrix[0][1]}</td>
                    </tr>

                    <tr>
                        <th>Actual: Leave (1)</th>
                        <td>${matrix[1][0]}</td>
                        <td>${matrix[1][1]}</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <p class="text-muted small">
            Rows represent actual classes and columns represent predicted classes.
        </p>
    `;

    document.getElementById("confusionMatrix").innerHTML = html;
}


// =====================================================
// START
// =====================================================

document.addEventListener("DOMContentLoaded", loadEvaluation);
