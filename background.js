/**
 * Deepfake Detector Background Script
 * Handles background processes like notifications and model loading
 */

// Global statistics storage
let stats = {
    totalScans: 0,
    totalDetections: 0,
    falsePositives: 0,
    lastScanTime: null
  };
  
  // Track each tab's detections
  let tabDetections = {};
  
  // Initialize extension
  chrome.runtime.onInstalled.addListener(async (details) => {
    if (details.reason === "install") {
      console.log("Deepfake Detector extension installed");
      
      // Initialize storage with default settings
      const defaultSettings = {
        sensitivity: 5,
        autoScan: false,
        showWarnings: true,
        showNotifications: true
      };
      
      chrome.storage.local.set({
        settings: defaultSettings,
        stats: stats
      });
      
      // Show welcome notification
      chrome.notifications.create({
        type: 'basic',
        iconUrl: 'icons/icon128.png',
        title: 'Deepfake Detector Installed',
        message: 'Click the extension icon to start detecting deepfakes on websites.',
        priority: 2
      });
    }
    
    // Set up badge text
    chrome.action.setBadgeBackgroundColor({ color: '#FF5252' });
  });
  
  // Listen for messages from popup or content scripts
  chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    const tabId = sender.tab ? sender.tab.id : null;
    
    switch (message.action) {
      case "scanResults":
        // Store results for the tab
        if (tabId) {
          tabDetections[tabId] = message.results;
          
          // Update badge with count
          updateBadge(tabId, message.results.length);
          
          // Update global stats
          stats.totalScans++;
          stats.totalDetections += message.results.length;
          stats.lastScanTime = new Date().toISOString();
          chrome.storage.local.set({ stats });
        }
        break;
        
      case "showNotification":
        // Show notification about detected deepfakes
        if (message.count > 0) {
          chrome.notifications.create({
            type: 'basic',
            iconUrl: 'icons/icon128.png',
            title: 'Deepfake Detected',
            message: `Found ${message.count} potential deepfake${message.count > 1 ? 's' : ''} on this page.`,
            priority: 2
          });
        }
        break;
        
      case "reportFalsePositive":
        // Record false positive feedback
        stats.falsePositives++;
        chrome.storage.local.set({ stats });
        
        // In a real extension, you might send this feedback to a server
        console.log("False positive reported:", message.imageUrl);
        break;
        
      case "getStats":
        sendResponse(stats);
        break;
    }
  });
  
  // Update badge when a tab is activated
  chrome.tabs.onActivated.addListener((activeInfo) => {
    const tabId = activeInfo.tabId;
    if (tabDetections[tabId]) {
      updateBadge(tabId, tabDetections[tabId].length);
    } else {
      chrome.action.setBadgeText({ text: "" });
    }
  });
  
  // Clear badge when a tab is closed
  chrome.tabs.onRemoved.addListener((tabId) => {
    if (tabDetections[tabId]) {
      delete tabDetections[tabId];
    }
  });
  
  // Update the badge with detection count
  function updateBadge(tabId, count) {
    if (count > 0) {
      chrome.action.setBadgeText({
        text: count.toString(),
        tabId: tabId
      });
    } else {
      chrome.action.setBadgeText({
        text: "",
        tabId: tabId
      });
    }
  }