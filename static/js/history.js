document.addEventListener('DOMContentLoaded', function() {
    // Initialize variables
    const predictionFilter = document.getElementById('predictionFilter');
    const dateFilter = document.getElementById('dateFilter');
    const searchFilter = document.getElementById('searchFilter');
    const predictionsGrid = document.getElementById('predictionsGrid');
    const emptyState = document.getElementById('emptyState');

    // Filter predictions
    function filterPredictions() {
        const predictionValue = predictionFilter.value.toLowerCase();
        const dateValue = dateFilter.value;
        const searchValue = searchFilter.value.toLowerCase();

        const predictionCards = predictionsGrid.querySelectorAll('.prediction-card');
        let visibleCount = 0;

        predictionCards.forEach(card => {
            const prediction = card.querySelector('.prediction-badge').textContent.toLowerCase();
            const date = card.querySelector('.detail-item:nth-child(2) .value').textContent;
            const recommendations = Array.from(card.querySelectorAll('.recommendations li'))
                .map(li => li.textContent.toLowerCase());

            const matchesPrediction = predictionValue === 'all' || prediction.includes(predictionValue);
            const matchesDate = !dateValue || date.includes(dateValue);
            const matchesSearch = !searchValue || 
                prediction.includes(searchValue) || 
                recommendations.some(rec => rec.includes(searchValue));

            if (matchesPrediction && matchesDate && matchesSearch) {
                card.parentElement.style.display = '';
                visibleCount++;
            } else {
                card.parentElement.style.display = 'none';
            }
        });

        // Show/hide empty state
        if (visibleCount === 0) {
            predictionsGrid.style.display = 'none';
            emptyState.classList.remove('d-none');
        } else {
            predictionsGrid.style.display = '';
            emptyState.classList.add('d-none');
        }
    }

    // Add event listeners
    predictionFilter.addEventListener('change', filterPredictions);
    dateFilter.addEventListener('change', filterPredictions);
    searchFilter.addEventListener('input', filterPredictions);

    // Initialize filters
    filterPredictions();

    // Add animation to cards when they come into view
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('animate__animated', 'animate__fadeInUp');
            }
        });
    }, { threshold: 0.1 });

    document.querySelectorAll('.prediction-card').forEach(card => {
        observer.observe(card);
    });

    // Add hover effects to prediction cards
    document.querySelectorAll('.prediction-card').forEach(card => {
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-5px)';
            this.style.boxShadow = '0 10px 20px rgba(0,0,0,0.2)';
        });

        card.addEventListener('mouseleave', function() {
            this.style.transform = '';
            this.style.boxShadow = '';
        });
    });

    // Add loading animation for filters
    const filters = document.querySelector('.filters');
    filters.addEventListener('change', function() {
        predictionsGrid.classList.add('loading');
        setTimeout(() => {
            predictionsGrid.classList.remove('loading');
        }, 500);
    });
}); 