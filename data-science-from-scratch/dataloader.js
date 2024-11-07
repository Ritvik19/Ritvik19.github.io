function create_nav_item(idx, content) {
  return `<li><a href="#line_${idx}">${content}</a></li>`;
}

function create_section(id, title, contents) {
  return `
    <section id="${id}" class="section">
        <div class="row">
            <div class="col-md-12 left-align">
                <h2 class="dark-text">
                    ${title}
                    <hr />
                </h2>
            </div>
        </div>

        <div class="row">
            <div class="col-md-12">
                ${contents}
            </div>
        </div>
    </section>`;
}

function create_table(contents) {
  return `
  <table class="table table-striped table-bordered">
  <tbody>
  ${contents}
  </tbody>
  </table>
  `;
}

function create_table_contents(data) {
  console.log(data);
  let contents = "";
  for (var i = 0; i < data.length; i++) {
    console.log(data[i]);
    contents += create_table_row(data[i].Title, data[i].Link);
  }
  return contents;
}

function create_table_row(title, link) {
  console.log(title, link);
  return `
  <tr>
  <td><a target="_blank" href="${link}">${title}</a></td>
  </tr>
  `;
}

let nav = document.getElementById("nav");
let container = document.getElementById("container");

for (var i = 0; i < notebook_categories.length; i++) {
  nav.innerHTML += create_nav_item(i + 1, notebook_categories[i]);
  container.innerHTML += create_section(
    `line_${i + 1}`,
    notebook_categories[i],
    create_table(create_table_contents(notebooks_data[i]))
  );
}
