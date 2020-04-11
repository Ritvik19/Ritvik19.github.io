var skill_div = document.getElementById('skills-contents')
var i = 0;
var skill_contents = "";
while (i <= 8) {
    skill_contents +=
        '<p> ' + skills['Skill'][i] + '</p>' +
        '<div class="w3-light-grey w3-round-xlarge w3-small">' +
        '<div class="w3-container w3-center w3-round-xlarge w3-teal" style="width:' +
        skills['Profeciency'][i] + '%">' + skills['Profeciency'][i] + '</div>' +
        '</div>'
    i += 1;
    console.log('inside the loop')
}
skill_div.innerHTML = skill_contents;

var certifications_div = document.getElementById('certifications-contents')
var i = 7;
var certification_contents = "";
while (i >= 0) {
    certification_contents +=
        '<div class="w3-container">' +
        '<a href="' + certifications['URL'][i] + '" target="_blank"><h5 class="w3-opacity" > <b>' + certifications['Certification'][i] + '</b></h5></a>' +
        '<h6 class="w3-text-teal"><i class="fa fa-calendar fa-fw w3-margin-right"></i>' + certifications['Date'][i] + '</h6>' +
        '<p>' + certifications['Authority'][i] + '</p> <br>' +
        '</div>'
    i -= 1;
}
certifications_div.innerHTML = certification_contents;

var portfolio_div = document.getElementById('portfolio-contents')
var i = 22;
var portfolio_contents = "";
while (i > 0) {
    portfolio_contents +=
        '<div class="card w3-margin">' +
        '<img src="img/' + i + '.png" style="width:100%">' +
        '<h1>' + portfolio['Title'][i] + '</h1>' +
        '<p>' + portfolio['Descriptions'][i] + '</p>'
    try {
        portfolio_contents += '<p><a href="' + portfolio['Buttons'][i][0]['URL'] + '" target="_blank"><button class="w3-teal w3-btn w3-hover">' + portfolio['Buttons'][i][0]['Text'] + '</button></a></p>'
        portfolio_contents += '<p><a href="' + portfolio['Buttons'][i][1]['URL'] + '" target="_blank"><button class="w3-teal w3-btn w3-hover">' + portfolio['Buttons'][i][1]['Text'] + '</button></a></p>'
        portfolio_contents += '<p><a href="' + portfolio['Buttons'][i][2]['URL'] + '" target="_blank"><button class="w3-teal w3-btn w3-hover">' + portfolio['Buttons'][i][2]['Text'] + '</button></a></p>'
    } catch {
        console.log(i)
    } finally {
        portfolio_contents += '</div>'
        i -= 1;
    }
}
portfolio_div.innerHTML = portfolio_contents;