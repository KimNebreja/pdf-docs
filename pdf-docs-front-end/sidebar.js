// Sidebar Functionality
const sidebar = document.getElementById('sidebar');
const mainContent = document.getElementById('mainContent');
const hamburger = document.getElementById('hamburger');
const externalToggle = document.getElementById('externalToggle');

// State Management
let isMobile = window.innerWidth <= 768;

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

// Event Listeners
hamburger.addEventListener('click', toggleSidebar);
externalToggle.addEventListener('click', toggleSidebar);
document.addEventListener('click', closeSidebarOnClickOutside);
window.addEventListener('resize', handleResize);

// Initial check
handleResize(); 