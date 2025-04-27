document.getElementById("scanButton").addEventListener("click", () => {
    document.getElementById("result").innerText = "Scanning ...";

    // Simulate process
    setTimeout(() => {
        document.getElementById("result").innerText = "Confidence: 85% Likely Real";  
    }, 2000); 
})