window.HELP_IMPROVE_VIDEOJS = false;

// More Works Dropdown Functionality
function toggleMoreWorks() {
    const dropdown = document.getElementById('moreWorksDropdown');
    const button = document.querySelector('.more-works-btn');
    
    if (dropdown.classList.contains('show')) {
        dropdown.classList.remove('show');
        button.classList.remove('active');
    } else {
        dropdown.classList.add('show');
        button.classList.add('active');
    }
}

// Close dropdown when clicking outside
document.addEventListener('click', function(event) {
    const container = document.querySelector('.more-works-container');
    const dropdown = document.getElementById('moreWorksDropdown');
    const button = document.querySelector('.more-works-btn');
    
    if (container && !container.contains(event.target)) {
        dropdown.classList.remove('show');
        button.classList.remove('active');
    }
});

// Close dropdown on escape key
document.addEventListener('keydown', function(event) {
    if (event.key === 'Escape') {
        const dropdown = document.getElementById('moreWorksDropdown');
        const button = document.querySelector('.more-works-btn');
        dropdown.classList.remove('show');
        button.classList.remove('active');
    }
});

// Copy BibTeX to clipboard
function copyBibTeX() {
    const bibtexElement = document.getElementById('bibtex-code');
    const button = document.querySelector('.copy-bibtex-btn');
    const copyText = button.querySelector('.copy-text');
    
    if (bibtexElement) {
        navigator.clipboard.writeText(bibtexElement.textContent).then(function() {
            // Success feedback
            button.classList.add('copied');
            copyText.textContent = 'Cop';
            
            setTimeout(function() {
                button.classList.remove('copied');
                copyText.textContent = 'Copy';
            }, 2000);
        }).catch(function(err) {
            console.error('Failed to copy: ', err);
            // Fallback for older browsers
            const textArea = document.createElement('textarea');
            textArea.value = bibtexElement.textContent;
            document.body.appendChild(textArea);
            textArea.select();
            document.execCommand('copy');
            document.body.removeChild(textArea);
            
            button.classList.add('copied');
            copyText.textContent = 'Cop';
            setTimeout(function() {
                button.classList.remove('copied');
                copyText.textContent = 'Copy';
            }, 2000);
        });
    }
}

// Scroll to top functionality
function scrollToTop() {
    window.scrollTo({
        top: 0,
        behavior: 'smooth'
    });
}

// Show/hide scroll to top button
window.addEventListener('scroll', function() {
    const scrollButton = document.querySelector('.scroll-to-top');
    if (window.pageYOffset > 300) {
        scrollButton.classList.add('visible');
    } else {
        scrollButton.classList.remove('visible');
    }
});

// Video carousel autoplay when in view
function setupVideoCarouselAutoplay() {
    const carouselVideos = document.querySelectorAll('.results-carousel video');
    
    if (carouselVideos.length === 0) return;
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            const video = entry.target;
            if (entry.isIntersecting) {
                // Video is in view, play it
                video.play().catch(e => {
                    // Autoplay failed, probably due to browser policy
                    console.log('Autoplay prevented:', e);
                });
            } else {
                // Video is out of view, pause it
                video.pause();
            }
        });
    }, {
        threshold: 0.5 // Trigger when 50% of the video is visible
    });
    
    carouselVideos.forEach(video => {
        observer.observe(video);
    });
}

// Video modal functionality
function setupVideoModal() {
    // Create modal element
    const modal = document.createElement('div');
    modal.className = 'video-modal';
    modal.innerHTML = `
        <div class="video-modal-content">
            <button class="video-modal-close" aria-label="Close video">
                <i class="fas fa-times"></i>
            </button>
            <video controls autoplay muted loop></video>
        </div>
    `;
    document.body.appendChild(modal);
    
    const modalVideo = modal.querySelector('video');
    const closeBtn = modal.querySelector('.video-modal-close');
    
    // Open modal on video click
    const videoItems = document.querySelectorAll('.media-video');
    videoItems.forEach(item => {
        item.addEventListener('click', function() {
            // Get video source from the video element itself
            const videoElement = this.querySelector('video source');
            if (videoElement && videoElement.src) {
                modalVideo.src = videoElement.src;
                modal.classList.add('active');
                // Play video
                modalVideo.play().catch(e => {
                    console.log('Video play failed:', e);
                });
            } else {
                // Fallback to data-video attribute
                const videoSrc = this.getAttribute('data-video');
                if (videoSrc) {
                    modalVideo.src = videoSrc;
                    modal.classList.add('active');
                    modalVideo.play().catch(e => {
                        console.log('Video play failed:', e);
                    });
                }
            }
        });
    });
    
    // Close modal
    function closeModal() {
        modal.classList.remove('active');
        modalVideo.pause();
        modalVideo.src = '';
    }
    
    closeBtn.addEventListener('click', closeModal);
    modal.addEventListener('click', function(e) {
        if (e.target === modal) {
            closeModal();
        }
    });
    
    // Close on Escape key
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape' && modal.classList.contains('active')) {
            closeModal();
        }
    });
}

// Setup video autoplay for media gallery
function setupMediaGalleryVideoAutoplay() {
    const galleryVideos = document.querySelectorAll('.media-gallery .media-video video');
    
    if (galleryVideos.length === 0) return;
    
    // Fix video source URLs to handle special characters in filenames
    galleryVideos.forEach(video => {
        const source = video.querySelector('source');
        if (source && source.getAttribute('src')) {
            const originalSrc = source.getAttribute('src');
            // Split path and encode only the filename
            const lastSlash = originalSrc.lastIndexOf('/');
            if (lastSlash !== -1) {
                const path = originalSrc.substring(0, lastSlash + 1);
                const filename = originalSrc.substring(lastSlash + 1);
                const encodedFilename = encodeURIComponent(filename);
                source.src = path + encodedFilename;
                video.load(); // Reload video with correct source
            }
        }
    });
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            const video = entry.target;
            if (entry.isIntersecting) {
                video.play().catch(e => {
                    console.log('Autoplay prevented:', e);
                });
            } else {
                video.pause();
            }
        });
    }, {
        threshold: 0.3
    });
    
    galleryVideos.forEach(video => {
        observer.observe(video);
    });
}

$(document).ready(function() {
    // Check for click events on the navbar burger icon

    var options = {
		slidesToScroll: 1,
		slidesToShow: 1,
		loop: true,
		infinite: true,
		autoplay: true,
		autoplaySpeed: 5000,
    }

	// Initialize all div with carousel class
    var carousels = bulmaCarousel.attach('.carousel', options);
	
    bulmaSlider.attach();
    
    // Setup video autoplay for carousel
    setupVideoCarouselAutoplay();
    
    // Setup media gallery video modal and autoplay
    setupVideoModal();
    setupMediaGalleryVideoAutoplay();

})
