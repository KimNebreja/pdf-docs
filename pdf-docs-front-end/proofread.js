// Proofreading Section Functionality
const proofreadSection = document.getElementById('proofreadSection');
const proofreadText = document.getElementById('proofreadText');
const downloadLink = document.getElementById('downloadLink');

// Check if we have proofread data
document.addEventListener('DOMContentLoaded', function() {
    const proofreadData = localStorage.getItem('proofreadData');
    if (proofreadData) {
        const data = JSON.parse(proofreadData);
        displayProofreadResult(data);
    } else {
        // If no data, redirect back to upload page
        window.location.href = 'index.html';
    }
});

function displayProofreadResult(data) {
    // Display proofread text
    proofreadText.value = data.proofread_text;
    proofreadSection.style.display = "block";

    // Set download link
    downloadLink.href = 'https://pdf-docs.onrender.com' + data.download_url;
    downloadLink.innerText = "Download Proofread DOCX";
    downloadLink.style.display = "block";
} 