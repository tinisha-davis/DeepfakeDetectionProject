/**
 * Deepfake Detector Content Script
 * This script runs in the context of web pages and handles the actual detection logic
 */

// Global variables
let model;
let isModelLoaded = false;
let isProcessing = false;
let settings = {
  sensitivity: 5,
  autoScan: false,
  showWarnings: true,
  showNotifications: true
};

// Initialize when extension is activated
async function initialize() {
  console.log("Initializing Deepfake Detector extension...");
  
  try {
    // Load settings from storage
    chrome.storage.local.get('settings', (data) => {
      if (data.settings) {
        settings = data.settings;
        console.log("Loaded settings:", settings);
      } else {
        // Save default settings
        chrome.storage.local.set({ settings });
      }
      
      // If auto-scan is enabled, scan the page immediately
      if (settings.autoScan) {
        setTimeout(scanImagesOnPage, 1500); // Give page time to fully load
      }
    });
    
    // Load required libraries and models
    await loadResources();
    
    console.log("Deepfake detection extension initialized successfully");
  } catch (error) {
    console.error("Error initializing deepfake detector:", error);
  }
}

// Load TensorFlow.js and face-api.js models
async function loadResources() {
  try {
    // Check if TensorFlow.js is loaded
    if (typeof tf === 'undefined') {
      console.error("TensorFlow.js is not loaded");
      return false;
    }
    
    // Check if face-api.js is loaded
    if (typeof faceapi === 'undefined') {
      console.error("face-api.js is not loaded");
      return false;
    }
    
    // Load face detection model
    await faceapi.nets.tinyFaceDetector.load('model/face-api-models');
    console.log("Face detection model loaded");
    
    // Load deepfake detection model
    model = await tf.loadLayersModel('model/deepfake_detector/model.json');
    isModelLoaded = true;
    console.log("Deepfake detection model loaded");
    
    return true;
  } catch (error) {
    console.error("Error loading resources:", error);
    return false;
  }
}

// Detect faces in an image
async function detectFaces(imageElement) {
  try {
    // Use face-api.js to detect faces
    const detections = await faceapi.detectAllFaces(
      imageElement, 
      new faceapi.TinyFaceDetectorOptions({ 
        inputSize: 416, 
        scoreThreshold: 0.5 
      })
    );
    
    return detections;
  } catch (error) {
    console.error("Error detecting faces:", error);
    return [];
  }
}

// Process image for prediction
async function preprocessImage(imageElement, faceDetection) {
  // Create a canvas to extract the face
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  
  // Get face box with some margin
  const { x, y, width, height } = faceDetection.box;
  const margin = Math.min(width, height) * 0.2; // 20% margin
  
  // Calculate dimensions with margin
  const faceX = Math.max(0, x - margin);
  const faceY = Math.max(0, y - margin);
  const faceWidth = width + margin * 2;
  const faceHeight = height + margin * 2;
  
  // Set canvas size to match model input requirements
  canvas.width = 224;
  canvas.height = 224;
  
  // Draw face on canvas
  ctx.drawImage(
    imageElement, 
    faceX, faceY, faceWidth, faceHeight,
    0, 0, 224, 224
  );
  
  // Convert to tensor and normalize
  return tf.tidy(() => {
    // Read pixel data
    const imgData = ctx.getImageData(0, 0, 224, 224);
    
    // Convert to tensor
    const tensor = tf.browser.fromPixels(imgData)
      .toFloat()
      .div(tf.scalar(255))
      .expandDims(0); // Add batch dimension
    
    return tensor;
  });
}

// Check if an image is large enough for processing
function isImageLargeEnough(imageElement) {
  const minDimension = 64; // Minimum width or height to consider
  return (
    imageElement.naturalWidth >= minDimension && 
    imageElement.naturalHeight >= minDimension
  );
}

// Check if an image is currently visible on screen
function isImageVisible(imageElement) {
  const rect = imageElement.getBoundingClientRect();
  
  // Check if image is in viewport
  return (
    rect.top < window.innerHeight &&
    rect.bottom > 0 &&
    rect.left < window.innerWidth &&
    rect.right > 0 &&
    getComputedStyle(imageElement).display !== 'none' &&
    getComputedStyle(imageElement).visibility !== 'hidden'
  );
}

// Mark an image as a potential deepfake
function markImageAsFake(imageElement, faceBox, confidence) {
  // Create a deepfake warning label
  const label = document.createElement('div');
  label.className = 'deepfake-warning';
  
  // Position the label
  const rect = imageElement.getBoundingClientRect();
  label.style.top = `${rect.top + window.scrollY + faceBox.y}px`;
  label.style.left = `${rect.left + window.scrollX + faceBox.x}px`;
  label.style.width = `${faceBox.width}px`;
  label.style.height = `${faceBox.height}px`;
  
  // Create warning content
  label.innerHTML = `
    <div class="deepfake-warning-icon">!</div>
    <div class="deepfake-warning-tooltip">
      <div class="deepfake-warning-header">Potential Deepfake Detected</div>
      <div class="deepfake-warning-confidence">${Math.round(confidence * 100)}% confidence</div>
      <div class="deepfake-warning-buttons">
        <button class="deepfake-warning-button">Details</button>
        <button class="deepfake-warning-button">Report</button>
      </div>
    </div>
  `;
  
  // Add to document
  document.body.appendChild(label);
  
  // Add a subtle red border to the image itself
  imageElement.style.outline = '3px solid rgba(255, 0, 0, 0.5)';
  
  // Store reference to the label on the image for later cleanup
  if (!imageElement.deepfakeLabels) {
    imageElement.deepfakeLabels = [];
  }
  imageElement.deepfakeLabels.push(label);
  
  // Add event listeners
  label.addEventListener('click', (event) => {
    event.stopPropagation();
    label.classList.toggle('deepfake-warning-expanded');
  });
  
  // Add buttons event listeners
  const buttons = label.querySelectorAll('.deepfake-warning-button');
  buttons[0].addEventListener('click', (event) => {
    event.stopPropagation();
    showDetailedAnalysis(imageElement, confidence);
  });
  
  buttons[1].addEventListener('click', (event) => {
    event.stopPropagation();
    reportFalsePositive(imageElement);
  });
  
  return label;
}

// Remove all deepfake warning labels
function clearDeepfakeWarnings() {
  // Remove all warning labels
  document.querySelectorAll('.deepfake-warning').forEach(label => {
    label.remove();
  });
  
  // Clear image outlines
  document.querySelectorAll('img').forEach(img => {
    if (img.deepfakeLabels) {
      img.style.outline = '';
      img.deepfakeLabels = null;
    }
  });
}

// Show detailed analysis popup
function showDetailedAnalysis(imageElement, confidence) {
  // In a real implementation, this would show more details about why the
  // image was classified as a deepfake
  console.log("Show detailed analysis for image:", imageElement, "confidence:", confidence);
  
  // Send message to open a detailed view in the popup
  chrome.runtime.sendMessage({
    action: "showDetailedAnalysis",
    imageUrl: imageElement.src,
    confidence: confidence
  });
}

// Report a false positive
function reportFalsePositive(imageElement) {
  console.log("Report false positive for image:", imageElement);
  
  // Send feedback to the background script
  chrome.runtime.sendMessage({
    action: "reportFalsePositive",
    imageUrl: imageElement.src
  });
  
  // Give user feedback
  alert("Thank you for your feedback. This helps improve our detection system.");
}

// Scan all images on the page
async function scanImagesOnPage() {
  if (!isModelLoaded || isProcessing) {
    console.log("Cannot scan: model not loaded or processing in progress");
    return;
  }
  
  isProcessing = true;
  console.log("Scanning page for potential deepfakes...");
  
  // Clear existing warnings
  clearDeepfakeWarnings();
  
  // Find all images on the page
  const images = document.querySelectorAll('img');
  console.log(`Found ${images.length} images on page`);
  
  // Track results
  const results = [];
  
  // Process each image
  for (const img of images) {
    // Only process visible, large enough images
    if (isImageVisible(img) && isImageLargeEnough(img)) {
      try {
        // Check if image is already loaded
        if (!img.complete) {
          continue; // Skip images that aren't loaded
        }
        
        // Detect faces in the image
        const faces = await detectFaces(img);
        
        // If no faces detected, skip
        if (!faces || faces.length === 0) {
          continue;
        }
        
        console.log(`Detected ${faces.length} faces in image:`, img.src);
        
        // Process each detected face
        for (const face of faces) {
          // Preprocess face for the model
          const tensor = await preprocessImage(img, face);
          
          // Make prediction
          const prediction = await model.predict(tensor);
          const fakeProbability = prediction.dataSync()[0];
          
          // Adjust based on sensitivity
          // Higher sensitivity means more likely to flag as fake
          const sensitivityAdjustment = (settings.sensitivity - 5) / 10; // -0.4 to 0.5
          const adjustedThreshold = 0.5 - sensitivityAdjustment;
          
          // Determine if it's a deepfake
          const isFake = fakeProbability > adjustedThreshold;
          
          if (isFake) {
            console.log("Potential deepfake detected:", fakeProbability);
            
            // Calculate confidence (higher probability = higher confidence)
            const confidence = fakeProbability;
            
            // Mark the image if warnings are enabled
            if (settings.showWarnings) {
              const label = markImageAsFake(img, face.box, confidence);
            }
            
            // Add to results
            results.push({
              imageUrl: img.src,
              imageDimensions: {
                width: img.naturalWidth,
                height: img.naturalHeight
              },
              faceLocation: {
                x: face.box.x,
                y: face.box.y,
                width: face.box.width,
                height: face.box.height
              },
              confidence: confidence,
              fakeProbability: fakeProbability,
              timestamp: new Date().toISOString()
            });
          }
          
          // Clean up tensor to prevent memory leaks
          tensor.dispose();
        }
      } catch (error) {
        console.error("Error processing image:", img.src, error);
      }
    }
  }
  
  // Send results to popup
  if (results.length > 0) {
    chrome.runtime.sendMessage({
      action: "scanResults",
      results: results
    });
    
    // Show notification if enabled
    if (settings.showNotifications) {
      chrome.runtime.sendMessage({
        action: "showNotification",
        count: results.length
      });
    }
  }
  
  console.log(`Scan complete. Found ${results.length} potential deepfakes.`);
  isProcessing = false;
  return results;
}

// Add a deepfake warning stylesheet
function addStyles() {
  const style = document.createElement('style');
  style.textContent = `
    .deepfake-warning {
      position: absolute;
      z-index: 9999;
      pointer-events: auto;
      box-sizing: border-box;
      cursor: pointer;
      border: 2px solid #ff5252;
      box-shadow: 0 0 0 2px rgba(255, 82, 82, 0.5);
      transition: all 0.2s ease;
    }
    
    .deepfake-warning::before {
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      background-color: rgba(255, 82, 82, 0.2);
    }
    
    .deepfake-warning-icon {
      position: absolute;
      top: -12px;
      right: -12px;
      width: 24px;
      height: 24px;
      border-radius: 50%;
      background-color: #ff5252;
      color: white;
      display: flex;
      align-items: center;
      justify-content: center;
      font-weight: bold;
      font-size: 16px;
      box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
      pointer-events: none;
    }
    
    .deepfake-warning-tooltip {
      position: absolute;
      top: calc(100% + 10px);
      left: 50%;
      transform: translateX(-50%) scale(0.9);
      opacity: 0;
      width: 200px;
      background-color: white;
      border-radius: 4px;
      box-shadow: 0 3px 8px rgba(0, 0, 0, 0.2);
      padding: 10px;
      pointer-events: none;
      transition: all 0.2s ease;
      z-index: 10000;
    }
    
    .deepfake-warning:hover .deepfake-warning-tooltip,
    .deepfake-warning-expanded .deepfake-warning-tooltip {
      opacity: 1;
      transform: translateX(-50%) scale(1);
      pointer-events: auto;
    }
    
    .deepfake-warning-header {
      color: #ff5252;
      font-weight: bold;
      font-size: 14px;
      margin-bottom: 5px;
      text-align: center;
    }
    
    .deepfake-warning-confidence {
      color: #333;
      font-size: 12px;
      margin-bottom: 8px;
      text-align: center;
    }
    
    .deepfake-warning-buttons {
      display: flex;
      justify-content: space-between;
    }
    
    .deepfake-warning-button {
      background-color: #f0f0f0;
      border: none;
      padding: 5px 10px;
      border-radius: 3px;
      font-size: 12px;
      cursor: pointer;
      flex: 1;
      margin: 0 3px;
    }
    
    .deepfake-warning-button:hover {
      background-color: #e0e0e0;
    }
  `;
  
  document.head.appendChild(style);
}

// Add an observer to detect new images loaded after initial scan
function setupMutationObserver() {
  // Create an observer instance
  const observer = new MutationObserver((mutations) => {
    let newImagesAdded = false;
    
    // Check if new images were added
    for (const mutation of mutations) {
      if (mutation.type === 'childList') {
        const images = mutation.target.querySelectorAll('img');
        if (images.length > 0) {
          newImagesAdded = true;
          break;
        }
      }
    }
    
    // If auto-scan is enabled and new images were added, scan the page
    if (settings.autoScan && newImagesAdded && !isProcessing) {
      // Wait a bit for images to load
      setTimeout(scanImagesOnPage, 500);
    }
  });
  
  // Observe the document body for added nodes
  observer.observe(document.body, { 
    childList: true, 
    subtree: true 
  });
  
  return observer;
}

// Handle messages from popup or background script
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  switch (message.action) {
    case "scan":
      scanImagesOnPage().then(results => {
        sendResponse({ status: "complete", count: results.length });
      });
      return true; // Indicates async response
      
    case "updateSettings":
      settings = message.settings;
      console.log("Updated settings:", settings);
      
      // Update storage
      chrome.storage.local.set({ settings });
      
      // If warnings were toggled off, clear all warnings
      if (!settings.showWarnings) {
        clearDeepfakeWarnings();
      }
      sendResponse({ status: "updated" });
      break;
      
    case "getStatus":
      sendResponse({
        isModelLoaded,
        isProcessing,
        settings
      });
      break;
  }
});

// Initialize when the page loads
window.addEventListener('load', () => {
  // Add custom CSS
  addStyles();
  
  // Initialize the detector
  initialize();
  
  // Set up observer for dynamically loaded content
  const observer = setupMutationObserver();
});

// Listen for visibility changes to handle tab switching
document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'visible' && settings.autoScan && isModelLoaded && !isProcessing) {
    // Run a scan when the tab becomes visible again
    setTimeout(scanImagesOnPage, 500);
  }
});