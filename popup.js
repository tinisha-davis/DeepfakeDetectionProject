document.addEventListener('DOMContentLoaded', function() {
    const scanButton = document.getElementById("scanButton");
    const resultElement = document.getElementById("result");

    if (scanButton && resultElement) {
        scanButton.addEventListener('click', function() {
            // this should indicate scanning has started
            resultElement.textContent = "Simulating Scan...";
            resultElement.classList.remove("high-alert", "moderate-alert", "low-alert", "error-alert");
            // simulation
            setTimeout(() => {
                const randomNum = Math.random();
                let probability;
                let alertLevel;
                let alertColorClass;
                // simulate high alert
                if (randomNum < 0.3) {
                    probability = (Math.random() * 30).toFixed(2);
                    alertLevel = "High";
                    alertColorClass = "high-alert";
                // simulate moderate alert    
                } else if (randomNum < 0.7) {
                    probability = (30 + Math.random() * 40).toFixed(2);
                    alertLevel = "Moderate";
                    alertColorClass - "moderate-alert";
                // simulate low alert
                } else if (randomNum < 0.95) {
                    probability = (70 + Math.random() * 30).toFixed(2);
                    alertLevel = "Low";
                    alertColorClass= "low-alert";
                // simulate error
                } else {
                    resultElement.textContent = "Error: Could not complete scan.";
                    resultElement.classList.add("error-alert");
                    return;
                }

                resultElement.textContent = "Confidence Rating: ${probability}% | Alert Level ${alertLevel}";
                resultElement.classList.add(alertColorClass);
            }, 1500)
        })
        
    } else {
        console.error("Scan button or result element not found in popup.html");
    }
})
