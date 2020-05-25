chart_options = {
    responsive: true,
    legend: {
        labels: {
            fontColor: "white",
        }
    },
    scale: {
        ticks: {
            suggestedMin: 0,
            suggestedMax: 100,
            fontColor: "white",
            backdropColor: "rgba(0,0,0,0)"
        },
        gridLines: {
            color: "gray"
        },
    }
}


var ctx1 = document.getElementById('chart1').getContext('2d');
var chart1 = new Chart(ctx1, {
    type: 'radar',
    data: {
        labels: ['HTML', 'CSS', 'JavaScript', 'CSS Frameworks', 'Flask'],
        datasets: [{
            label: 'Web Development',
            backgroundColor: 'rgba(89, 24, 223, 0.5)',
            borderColor: 'rgb(89, 24, 223)',
            data: [90, 90, 85, 85, 75]
        }]
    },
    options: chart_options,
});

var ctx2 = document.getElementById('chart2').getContext('2d');
var chart2 = new Chart(ctx2, {
    type: 'radar',
    data: {
        labels: ['Python', 'Data Science', 'Machine Learning', 'Deep Learning', 'Web Scraping'],
        datasets: [{
            label: 'Data Science',
            backgroundColor: 'rgba(89, 24, 223, 0.5)',
            borderColor: 'rgb(89, 24, 223)',
            data: [90, 80, 80, 65, 85]
        }]
    },
    options: chart_options,
});

var ctx3 = document.getElementById('chart3').getContext('2d');
var chart3 = new Chart(ctx3, {
    type: 'radar',
    data: {
        labels: ['Java (Core)', 'C++', 'Bash Scripting', 'C', 'DS Algo'],
        datasets: [{
            label: 'Other',
            backgroundColor: 'rgba(89, 24, 223, 0.5)',
            borderColor: 'rgb(89, 24, 223)',
            data: [70, 70, 80, 75, 70]
        }]
    },
    options: chart_options,
});