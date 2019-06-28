function webCam() {
    eel.webCam();
}

function configs() {
    let weightsPath = document.getElementById("weightsPath").value;
    let outputPath = document.getElementById("outputPath").value;
    let confidence = document.getElementById("confidence").value;
    let threshold = document.getElementById("threshold").value;
    let checkBox = document.getElementById("showTime");
    let showTime = checkBox.checked;
    eel.update_params(weightsPath, outputPath, confidence, threshold, showTime);

}