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
  <thead>
  <tr>
  <th>Title</th>
  <th>Date</th>
  <th>Description</th>
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
      data[i].title,
      data[i].link,
      data[i].date,
      data[i].description
    );
  }
  return contents;
}

function create_table_row(title, link, date, description) {
  console.log(title, link, date, description);
  return `
  <tr>
  <td><a target="_blank" href="${link}">${title}</a></td>
  <td>${date}</td>
  <td>${description}</td>
  </tr>
  `;
}

function create_literature_review_section_contents(
  header,
  literature_review_data
) {
  let contents = `<h4>${header}</h4><table class="table table-striped table-bordered">`;
  for (var i = 0; i < literature_review_data.length; i++) {
    contents += `<tr><td><a target="_blank" href="${literature_review_data[i].link}">${literature_review_data[i].title}</a></td></tr>`;
  }
  contents += "</table>";
  return contents;
}

let nav = document.getElementById("nav");
let container = document.getElementById("container");
for (var i = 0; i < nav_data.length; i++) {
  nav.innerHTML += create_nav_item(i + 1, nav_data[i]);
  container.innerHTML += create_section(
    `line_${i + 1}`,
    nav_data[i],
    create_table(create_table_contents(papers_data[i]))
  );
}

nav.innerHTML += create_nav_item(
  nav_data.length + 1,
  "Literature Review and Reading Lists"
);
container.innerHTML += create_section(
  `line_${nav_data.length + 1}`,
  "Literature Reviews and Reading Lists",
  create_literature_review_section_contents(
    "Literature Reviews",
    literature_review_data
  ) +
    create_literature_review_section_contents(
      "Reading Lists",
      reading_list_data
    )
);
