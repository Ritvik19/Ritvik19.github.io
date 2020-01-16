var skill_div = document.getElementById('skills-contents')
var i = 0;
var skill_contents = "";
while (i <= 9) {
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
var i = 6;
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