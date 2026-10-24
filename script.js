const imageInput = document.getElementById("imageInput");
const uploadButton = document.getElementById("uploadButton");
const cameraButton = document.getElementById("cameraButton");
const cameraPanel = document.getElementById("cameraPanel");
const cameraVideo = document.getElementById("cameraVideo");
const captureButton = document.getElementById("captureButton");
const closeCameraButton = document.getElementById("closeCameraButton");
const canvas = document.createElement("canvas");
const previewPanel = document.getElementById("previewPanel");
const imagePreview = document.getElementById("imagePreview");
const analyzeButton = document.getElementById("analyzeButton");
const removeButton = document.getElementById("removeButton");
const statusMessage = document.getElementById("statusMessage");
const resultsCard = document.getElementById("resultsCard");

let selectedFile = null;
let cameraStream = null;
let previewUrl = null;

function setStatus(message, type = "") {
  statusMessage.textContent = message;
  statusMessage.className = `status ${type}`;
}

function stopCamera() {
  if (cameraStream) {
    cameraStream.getTracks().forEach(track => track.stop());
    cameraStream = null;
  }
  cameraVideo.srcObject = null;
  cameraPanel.hidden = true;
}

function setSelectedImage(file) {
  if (!file || !file.type.startsWith("image/")) {
    setStatus("Please choose a valid image file.", "error");
    return;
  }
  if (file.size > 10 * 1024 * 1024) {
    setStatus("Image is too large. Maximum size is 10 MB.", "error");
    return;
  }
  selectedFile = file;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  imagePreview.src = previewUrl;
  previewPanel.hidden = false;
  resultsCard.hidden = true;
  setStatus("Image ready. Select Analyze Leaf to continue.", "success");
}

uploadButton.addEventListener("click", () => imageInput.click());
imageInput.addEventListener("change", () => {
  if (imageInput.files && imageInput.files[0]) setSelectedImage(imageInput.files[0]);
});

cameraButton.addEventListener("click", async () => {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    setStatus("Camera access is unavailable in this browser. Try uploading an image or use HTTPS/localhost.", "error");
    return;
  }
  try {
    stopCamera();
    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: "environment" } },
      audio: false
    });
    cameraVideo.srcObject = cameraStream;
    cameraPanel.hidden = false;
    setStatus("Camera is ready. Position a tomato leaf and capture a photo.");
  } catch (error) {
    setStatus("Could not open the camera. Allow camera permission or upload an image instead.", "error");
  }
});

captureButton.addEventListener("click", () => {
  if (!cameraVideo.videoWidth || !cameraVideo.videoHeight) {
    setStatus("Camera is still starting. Please try again in a moment.", "error");
    return;
  }
  canvas.width = cameraVideo.videoWidth;
  canvas.height = cameraVideo.videoHeight;
  canvas.getContext("2d").drawImage(cameraVideo, 0, 0);
  canvas.toBlob(blob => {
    if (!blob) {
      setStatus("Could not capture the image. Please try again.", "error");
      return;
    }
    const file = new File([blob], "tomato-leaf-capture.jpg", { type: "image/jpeg" });
    setSelectedImage(file);
    stopCamera();
  }, "image/jpeg", 0.92);
});

closeCameraButton.addEventListener("click", stopCamera);

removeButton.addEventListener("click", () => {
  selectedFile = null;
  imageInput.value = "";
  previewPanel.hidden = true;
  resultsCard.hidden = true;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
  imagePreview.removeAttribute("src");
  setStatus("Image cleared.");
});

function addListItems(elementId, items) {
  const list = document.getElementById(elementId);
  list.replaceChildren();
  (items || []).forEach(item => {
    const li = document.createElement("li");
    li.textContent = item;
    list.appendChild(li);
  });
}

analyzeButton.addEventListener("click", async () => {
  if (!selectedFile) {
    setStatus("Please capture or choose an image first.", "error");
    return;
  }

  const formData = new FormData();
  formData.append("image", selectedFile);
  analyzeButton.disabled = true;
  analyzeButton.textContent = "Analyzing…";
  setStatus("Analyzing image with the CNN model. Please wait.");
  resultsCard.hidden = true;

  try {
    const response = await fetch("/predict", { method: "POST", body: formData });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Analysis failed.");

    document.getElementById("predictionName").textContent = data.prediction;
    document.getElementById("confidenceValue").textContent = `${data.confidence}%`;
    document.getElementById("predictionDescription").textContent = data.description;
    document.getElementById("notice").textContent = data.notice;
    addListItems("treatmentList", data.treatment);
    addListItems("preventionList", data.prevention);
    resultsCard.hidden = false;
    resultsCard.scrollIntoView({ behavior: "smooth", block: "start" });
    setStatus("Analysis complete.", "success");
  } catch (error) {
    setStatus(error.message || "Could not connect to the TomatoCare server.", "error");
  } finally {
    analyzeButton.disabled = false;
    analyzeButton.textContent = "Analyze leaf";
  }
});

window.addEventListener("beforeunload", () => {
  stopCamera();
  if (previewUrl) URL.revokeObjectURL(previewUrl);
});
