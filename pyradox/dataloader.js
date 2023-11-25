function create_nav_item(idx, content) {
  return `<li><a href="#line_${idx}">${content}</a></li>`;
}

function create_code_block(code) {
  return `<pre class="brush: python;">${code}</pre>`;
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
  <thead>
  <tr>
  <th>Module</th>
  <th>Description</th>
  <th>Input Shape</th>
  <th>Output Shape</th>
  </tr>
  </thead>
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
    contents += create_table_row(
      data[i]["Module"],
      data[i]["Usage"],
      data[i]["Description"],
      data[i]["Input Shape"],
      data[i]["Output Shape"]
    );
  }
  return contents;
}

function create_table_row(title, link, description, input_shape, output_shape) {
  console.log(title, link, description, input_shape, output_shape);
  return `
  <tr>
  <td><a target="_blank" href="${link}">${title}</a></td>
  <td>${description}</td>
  <td>${input_shape}</td>
  <td>${output_shape}</td>
  </tr>
  `;
}

let nav = document.getElementById("nav");
let container = document.getElementById("container");

nav.innerHTML += create_nav_item(1, "Installation");
container.innerHTML += create_section(
  "line_1",
  "Installation",
  create_code_block("pip install pyradox-generative")
);

for (var i = 0; i < nav_data.length; i++) {
  nav.innerHTML += create_nav_item(i + 2, nav_data[i]);
  container.innerHTML += create_section(
    `line_${i + 2}`,
    nav_data[i],
    create_table(create_table_contents(usage_data[i]))
  );
}
