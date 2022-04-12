function get_nav_item(index, title) {
  return `<li><a href="#line3_${index}">${title}</a></li>`;
}

function get_github_user(github) {
  if (github !== undefined) return github;
  return "Ritvik19";
}

function get_documentation(title, no_documentation) {
  if (no_documentation !== undefined) return "";
  return `<p><a href="/${title}" role="button">View Documentation</a></p>`;
}

function get_container_block(
  index,
  title,
  description,
  github,
  no_documentation
) {
  return `
        <div class="row">
            <div class="col-md-12 left-align">
                <h4 class="dark-text" id="line3_${index}">
                    ${title}
                </h4>
            </div>
            <div class="col-md-12">
                <p>${description}</p>
                <p><a href="https://github.com/${get_github_user(
                  github
                )}/${title}" role="button" target="_blank">View Project</a></p>
                ${get_documentation(title, no_documentation)}
            </div>
        </div>`;
}

function get_skill_bar(skill, progress) {
  return `
    <div class="progress">
        <div class="progress-bar bg-info" role="progressbar" style="width: ${progress}%" aria-valuenow="${progress}" aria-valuemin="0" aria-valuemax="100">
            ${skill}
        </div>
    </div>`;
}

let nav = document.getElementById("nav");
let container = document.getElementById("container");
let skill_set = document.getElementById("skills");

for (let i = 0; i < data.length; i++) {
  nav.innerHTML += get_nav_item(i + 1, data[i]["title"]);
  container.innerHTML += get_container_block(
    i + 1,
    data[i]["title"],
    data[i]["description"],
    data[i]["github"],
    data[i]["no_documentation"]
  );
}

for (let s in skills) {
  skill_set.innerHTML += get_skill_bar(s, skills[s]);
}
