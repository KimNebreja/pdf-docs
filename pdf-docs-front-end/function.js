       // DOM Elements
       const sidebar = document.getElementById('sidebar');
       const mainContent = document.getElementById('mainContent');
       const hamburger = document.getElementById('hamburger');
       const externalToggle = document.getElementById('externalToggle');
       const uploadForm = document.getElementById('uploadForm');
       const fileInput = document.getElementById('pdfFile');
       const status = document.getElementById('status');
       const proofreadSection = document.getElementById('proofreadSection');
       const proofreadText = document.getElementById('proofreadText');
       const downloadLink = document.getElementById('downloadLink');
       const uploadButton = document.querySelector('.upload-button');

       // State Management
       let isMobile = window.innerWidth <= 768;
       let isUploading = false;

       // Sidebar Functionality
       function toggleSidebar() {
           if (isMobile) {
               sidebar.classList.toggle('active');
               mainContent.classList.toggle('expanded');
           } else {
               sidebar.classList.toggle('collapsed');
               mainContent.classList.toggle('expanded');
           }
       }

       function closeSidebarOnClickOutside(e) {
           if (isMobile && sidebar.classList.contains('active') && 
               !sidebar.contains(e.target) && 
               !externalToggle.contains(e.target)) {
               sidebar.classList.remove('active');
               mainContent.classList.remove('expanded');
           }
       }

       function handleResize() {
           const newIsMobile = window.innerWidth <= 768;
           if (newIsMobile !== isMobile) {
               isMobile = newIsMobile;
               if (!isMobile) {
                   sidebar.classList.remove('active');
                   mainContent.classList.remove('expanded');
               }
           }
       }

       // File Upload Functionality
       function validateFile(file) {
           if (!file) {
               updateStatus("Please select a file", true);
               return false;
           }
           if (file.type !== 'application/pdf') {
               updateStatus("Please upload a PDF file", true);
               return false;
           }
           if (file.size > 10 * 1024 * 1024) { // 10MB limit
               updateStatus("File size should be less than 10MB", true);
               return false;
           }
           return true;
       }

       function updateStatus(message, isError = false) {
           status.innerText = message;
           status.style.color = isError ? '#ff4444' : '#332219';
           uploadForm.classList.toggle('uploading', !isError);
       }

       function handleFiles(files) {
           if (files.length === 0) return;
           
           const file = files[0];
           if (!validateFile(file)) return;

           if (!isUploading) {
               handleFormSubmit();
           }
       }

       // Drag and Drop Functionality
       function preventDefaults(e) {
           e.preventDefault();
           e.stopPropagation();
       }

       function highlight(e) {
           uploadForm.classList.add('highlight');
       }

       function unhighlight(e) {
           uploadForm.classList.remove('highlight');
       }

       function handleDrop(e) {
           const dt = e.dataTransfer;
           const files = dt.files;
           fileInput.files = files;
           handleFiles(files);
       }

       // Form Submission
       async function handleFormSubmit() {
           if (isUploading) return;
           if (!fileInput.files.length) {
               updateStatus("Please select a file", true);
               return;
           }

           const file = fileInput.files[0];
           if (!validateFile(file)) return;

           isUploading = true;
           updateStatus("Uploading...");
           uploadForm.classList.add('uploading');
           uploadButton.disabled = true;

           try {
               const formData = new FormData();
               formData.append('file', file);

               const response = await fetch('https://pdf-docs.onrender.com/convert', { 
                   method: 'POST', 
                   body: formData 
               });

               if (!response.ok) {
                   throw new Error(`HTTP error! status: ${response.status}`);
               }

               const data = await response.json();
               
               // Display proofread text
               proofreadText.value = data.proofread_text;
               proofreadSection.style.display = "block";

               // Set download link
               downloadLink.href = 'https://pdf-docs.onrender.com' + data.download_url;
               downloadLink.innerText = "Download Proofread DOCX";
               downloadLink.style.display = "block";

               updateStatus("Upload successful!");
               
           } catch (error) {
               console.error('Upload error:', error);
               updateStatus("Error: " + error.message, true);
           } finally {
               isUploading = false;
               uploadForm.classList.remove('uploading');
               uploadButton.disabled = false;
           }
       }

       // Event Listeners
       hamburger.addEventListener('click', toggleSidebar);
       externalToggle.addEventListener('click', toggleSidebar);
       document.addEventListener('click', closeSidebarOnClickOutside);
       window.addEventListener('resize', handleResize);

       // File Upload Event Listeners
       ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
           uploadForm.addEventListener(eventName, preventDefaults, false);
       });

       ['dragenter', 'dragover'].forEach(eventName => {
           uploadForm.addEventListener(eventName, highlight, false);
       });

       ['dragleave', 'drop'].forEach(eventName => {
           uploadForm.addEventListener(eventName, unhighlight, false);
       });

       uploadForm.addEventListener('drop', handleDrop, false);
       fileInput.addEventListener('change', function() {
           handleFiles(this.files);
       });
       uploadButton.addEventListener('click', function() {
           fileInput.click();
       });

       // Initial check
       handleResize();