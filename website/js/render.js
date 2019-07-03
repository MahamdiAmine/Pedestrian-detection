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

let videoPath = '';

function video() {
    if (confirm("   Please wait this might take a while")) {
        if (videoPath) {
            eel.test("./data/" + videoPath);
        } else {
            console.log("Choose the video first !");
            alert("Choose the video first !");
        }
    }
}

function done() {
    alert("     [✔] Done.");
}


function readURL(input) {
    if (input.files && input.files[0]) {
        videoPath = input.files[0].name;
        $('.image-upload-wrap').hide();
        $('.file-upload-image').attr('src', 'images/done.png');
        $('.file-upload-content').show();
    } else {
        removeUpload();
    }
}

function removeUpload() {
    $('.file-upload-input').replaceWith($('.file-upload-input').clone());
    $('.file-upload-content').hide();
    $('.image-upload-wrap').show();
}

$('.image-upload-wrap').bind('dragover', function () {
    $('.image-upload-wrap').addClass('image-dropping');
});
$('.image-upload-wrap').bind('dragleave', function () {
    $('.image-upload-wrap').removeClass('image-dropping');
});
