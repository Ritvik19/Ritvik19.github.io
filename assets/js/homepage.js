function get_nav_item(index, title, super_index) {
  return `<li><a href="#line${super_index}_${index}">${title}</a></li>`;
}

function get_action_buttons(actions) {
  console.log(actions);
  columns = 2
  let rows = "";
  let cols = "";
  let count = 0;
  let table = "<table>"
  for (const element of actions) {
    console.log(element)
    if (count % columns == 0) {
      rows += "<tr>"
    }
    rows += `<td style="padding: 0px 10px;"><a href="${element["link"]}" role="button" target="_blank">${element["title"]}</a></td>`
    count += 1;
    if (count % columns == 0) {
      rows += "</tr>"
    }
  }
  table += rows
  table += "</table>"
  return table;
}

function get_project_block(
  index,
  title,
  description,
  actions,
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
                ${get_action_buttons(actions)}
            </div>
            <div class="col-md-12">
                <br><hr><br>
            </div>
        </div>`;
}

function get_space_block(index, title, description, linked_models) {
  return `
        <div class="row">
            <div class="col-md-12 left-align">
                <h6 class="dark-text" id="line4_${index}">
                    ${title}
                </h6>
            </div>
            <div class="col-md-12">
                <p>${description}</p>
                <p><a href="https://huggingface.co/spaces/Ritvik19/${title}" role="button" target="_blank">View Space</a></p>
                ${get_models_table(linked_models)}
            </div>
        </div>`;
}

function get_models_table(linked_models) {
  return `
    <table class="table table-striped">
        <thead>
            <tr>
                <th scope="col">Model</th>
                <th scope="col">Description</th>
            </tr>
        </thead>
        <tbody>
            ${get_models_table_rows(linked_models)}
        </tbody>
    </table>`;
}

function get_models_table_rows(linked_models) {
  let rows = "";
  for (const element of linked_models) {
    rows += ` <tr>  
                <td><a href="https://huggingface.co/Ritvik19/${element[0]}" target="_blank">${element[0]}</a></td>
                <td>${element[1]}</td>
            </tr>`;
  }
  return rows;
}

function get_skill_bar(skill, progress) {
  return `
    <div class="progress">
        <div class="progress-bar bg-info" role="progressbar" style="width: ${progress}%" aria-valuenow="${progress}" aria-valuemin="0" aria-valuemax="100">
            ${skill}
        </div>
    </div>`;
}

let nav_projects = document.getElementById("nav-projects");
let container_projects = document.getElementById("container-projects");
let nav_models = document.getElementById("nav-models");
let container_models = document.getElementById("container-models");
let skill_set = document.getElementById("skills");

for (let i = 0; i < data.length; i++) {
  nav_projects.innerHTML += get_nav_item(i + 1, data[i]["title"], 3);
  container_projects.innerHTML += get_project_block(
    i + 1,
    data[i]["title"],
    data[i]["description"],
    data[i]["actions"]
  );
}

for (let s in skills) {
  skill_set.innerHTML += get_skill_bar(s, skills[s]);
}

for (let i = 0; i < models.length; i++) {
  nav_models.innerHTML += get_nav_item(i + 1, models[i]["title"], 4);
  container_models.innerHTML += get_space_block(
    i + 1,
    models[i]["title"],
    models[i]["description"],
    models[i]["linked_models"]
  );
}
