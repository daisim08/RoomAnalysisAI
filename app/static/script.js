const imageInput = document.getElementById("imageInput");
const preview = document.getElementById("preview");
const analyzeButton = document.getElementById("analyzeButton");
const loading = document.getElementById("loading");
const resultCard = document.getElementById("resultCard");

// Show selected image
imageInput.addEventListener("change", function () {

    const file = this.files[0];

    if (!file) {
        preview.style.display = "none";
        analyzeButton.disabled = true;
        resultCard.style.display = "none";
        return;
    }

    const reader = new FileReader();

    reader.onload = function (e) {
        preview.src = e.target.result;
        preview.style.display = "block";
    };

    reader.readAsDataURL(file);

    analyzeButton.disabled = false;
});


// Analyze button
analyzeButton.addEventListener("click", analyzeRoom);

async function analyzeRoom() {

    const file = imageInput.files[0];

    if (!file) {
        alert("Please select an image.");
        return;
    }

    const formData = new FormData();
    formData.append("image", file);

    loading.style.display = "block";
    analyzeButton.disabled = true;
    resultCard.style.display = "none";

    try {

        const response = await fetch("/predict", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error);
        }

        document.getElementById("condition").textContent =
            data.class;

        document.getElementById("confidence").textContent =
            data.confidence + "%";

        document.getElementById("crackProb").textContent =
            data.probabilities.crack + "%";

        document.getElementById("dampnessProb").textContent =
            data.probabilities.dampness + "%";

        document.getElementById("normalProb").textContent =
            data.probabilities.normal + "%";

        document.getElementById("paintProb").textContent =
            data.probabilities.peeling_paint + "%";


        // Repair Priority
        let priority = "";
        let material = "";

        switch (data.class) {

            case "crack":
                priority = "🔴 High Priority";
                material = "Wall Putty, Cement, Crack Filler";
                break;

            case "dampness":
                priority = "🟠 Medium Priority";
                material = "Waterproof Paint, Damp Proof Coating";
                break;

            case "peeling_paint":
                priority = "🟡 Medium Priority";
                material = "Primer, Interior Emulsion Paint";
                break;

            default:
                priority = "🟢 No Repair Needed";
                material = "No material required";
        }

        document.getElementById("priority").textContent =
            priority;

        document.getElementById("material").textContent =
            material;

        resultCard.style.display = "block";

    }

    catch (error) {

        alert(error.message);

    }

    finally {

        loading.style.display = "none";
        analyzeButton.disabled = false;

    }

}