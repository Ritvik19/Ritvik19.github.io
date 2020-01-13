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