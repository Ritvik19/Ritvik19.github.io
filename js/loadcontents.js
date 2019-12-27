var skill_div = document.getElementById('skills-contents')
var i = 0;
var skill_contents = "";
while (i <= 9) {
    skill_contents += '<div class="w3-section">' +
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