var certifications_div = document.getElementById('certifications-contents')
var i = 7;
var certification_contents = "<li>[</li>";
while (i >= 0) {
    certification_contents +=
        '<li><span class="dim">··</span>{</li>' +
        '<li><span class="dim">····</span>"Title": <a href="' + certifications['URL'][i] + '" target="_blank" class="highlight">"' + certifications['Certification'][i] + '</a>", </li>' +
        '<li><span class="dim">····</span>"Completed": "' + certifications['Date'][i] + '",' +
        ' "Organization": "' + certifications['Authority'][i] + '", </li>' +
        '<li><span class="dim">··</span>},</li>'
    i -= 1;
}
certification_contents += '<li>]</li>'
certifications_div.innerHTML = certification_contents;

var portfolio_div = document.getElementById('portfolio-contents')
var i = 23;
var portfolio_contents = "";
while (i > 0) {
    portfolio_contents +=
        '<div class="card">' +
        '<img class="card-img-top" src = "img/' + i + '.png" alt = "error">' +
        '<div class="card-body">' +
        '<h4 class="card-title"> $ <span class="highlight">' + portfolio['Title'][i] + '</span></h4>' +
        '<p class="card-text">' + portfolio['Descriptions'][i] + '</p>'
    try {
        portfolio_contents += '<a href="' + portfolio['Buttons'][i][0]['URL'] + '" target="_blank" class="card-link">--' + portfolio['Buttons'][i][0]['Text'] + '</a>'
        portfolio_contents += '<a href="' + portfolio['Buttons'][i][1]['URL'] + '" target="_blank" class="card-link">--' + portfolio['Buttons'][i][1]['Text'] + '</a>'
        portfolio_contents += '<a href="' + portfolio['Buttons'][i][2]['URL'] + '" target="_blank" class="card-link">--' + portfolio['Buttons'][i][2]['Text'] + '</a>'
    } catch {
        console.log(i)
    } finally {
        portfolio_contents += '</div></div>'
        i -= 1;
    }
}
portfolio_div.innerHTML = portfolio_contents;

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