var skill_div = document.getElementById('skills-contents')
var i = 0;
var skill_contents = "";
while (i <= 9) {
    skill_contents +=
        '<div class="w3-section">' +
        '<p class="w3-wide"> ' + skills['Skill'][i] + '</p>' +
        '<div class="w3-white">' +
        '<div class="w3-blue" style="height:28px;width:' + skills['Profeciency'][i] + '%"></div>' +
        '</div>' +
        '<hr>' +
        '</div>'
    i += 1;
}
skill_div.innerHTML = skill_contents;

var certifications_div = document.getElementById('certifications-contents')
var i = 14;
var certification_contents = "";
while (i >= 0) {
    certification_contents +=
        '<div class="flip-card">' +
        '<div class="flip-card-inner">' +
        '<div class="flip-card-front w3-indigo">' +
        '<div>' +
        '<p>' + certifications['Date'][i] + '</p>' +
        '<h5>' + certifications['Certification'][i] + '</h5>' +
        '<p>' + certifications['Authority'][i] + '</p>' +
        '</div>' +
        '</div >' +
        '<div class="flip-card-back w3-indigo">' +
        '<a href="' + certifications['URL'][i] + '" target="_blank">' +
        '<button type="button" class="w3-button w3-round-large w3-cyan w3-hover-blue w3-ripple">View</button>' +
        '</a>' +
        '</div>' +
        '</div>' +
        '</div >'
    i -= 1;
}
certifications_div.innerHTML = certification_contents;

var portfolio_div = document.getElementById('portfolio-contents')
var i = 40;
var portfolio_contents = "";
while (i > 0) {
    portfolio_contents +=
        '<div class="column ' + portfolio['Classes'][i] + '">' +
        '<div class="content w3-indigo">' +
        '<img src="img/' + portfolio['Id'][i].replace('-', '') + '.png" alt="" style="width:100%">' +
        '<button type="button" class="btn btn-md btn-block w3-hover-blue" data-toggle="modal" data-target="#' + portfolio['Id'][i] + '">' + portfolio['Title'][i] + '</button>' +
        '<div class="modal fade" tabindex="-1" role="dialog" aria-hidden="true" id="' + portfolio['Id'][i] + '">' +
        '<div class="modal-dialog modal-lg">' +
        '<div class="modal-content">' +
        '<div class="modal-header">' +
        '<h5 class="modal-title">' + portfolio['Title'][i] + '</h5>' +
        '<button type="button" class="close" data-dismiss="modal" aria-label="Close">' +
        '<span aria-hidden="true">&times;</span>' +
        '</button>' +
        '</div>' +
        '<img src="img/' + portfolio['Id'][i].replace('-', '') + '.png" alt="" style="width:100%">' +
        '<p>' + portfolio['Descriptions'][i] + '</p>'
    try {
        portfolio_contents += '<a class="w3-hover-blue" href="' + portfolio['Buttons'][i][0]['URL'] + '" target="_blank">' + portfolio['Buttons'][i][0]['Text'] + '</a>'
        portfolio_contents += '<a class="w3-hover-blue" href="' + portfolio['Buttons'][i][1]['URL'] + '" target="_blank">' + portfolio['Buttons'][i][1]['Text'] + '</a>'
    } catch {
        console.log(i)
    } finally {
        portfolio_contents +=
            '</div>' +
            '</div>' +
            '</div>' +
            '</div>' +
            '</div>'
        i -= 1;
    }
}
portfolio_div.innerHTML = portfolio_contents;