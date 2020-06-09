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

var achievements_div = document.getElementById('achievements-contents')
var achievements_contents = "";
console.log(achievements)
var i = 0;
while (i < achievements.length) {
    console.log(achievements[i])
    achievements_contents += '<p class="w3-padding w3-panel achievement">' + achievements[i] + '</p>'
    i++;
}
console.log(achievements_contents)
achievements_div.innerHTML = achievements_contents;

var portfolio_div = document.getElementById('portfolio-contents')
var i = 23;
var portfolio_contents = "";
while (i > 0) {
    portfolio_contents +=
        '<div class="card project ' + portfolio['Class'][i] + '">' +
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