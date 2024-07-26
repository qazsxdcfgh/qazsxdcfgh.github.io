// Import ONNX.js
import * as onnx from 'https://cdn.jsdelivr.net/npm/onnxjs/dist/onnx.min.js';

// Load the ONNX model
const model = await onnx.load('onnx_model.onnx');

// Create a session
const session = new onnx.InferenceSession();

// Set the CPU backend
session.backend = 'cpu';

// Get the input and output elements
const imageInput = document.getElementById('imageInput');
const predictButton = document.getElementById('predictButton');
const resultDiv = document.getElementById('result');

// Add event listener to predict button
predictButton.addEventListener('click', async () => {
  // Get the selected image file
  const file = imageInput.files[0];

  // Read the image file as a tensor
  const tensor = await onnx.Tensor.fromFile(file, 'float32', [1, 28, 28]);

  // Run the model
  const outputs = await session.run([tensor]);

  // Get the predicted class index
  const predictedClass = outputs[0].data.indexOf(Math.max(...outputs[0].data));

  // Display the result
  resultDiv.innerText = `Predicted digit: ${predictedClass}`;
});
