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

function create_cards_container(contents) {
  return `
  <div class="card-columns">
  ${contents}
  </div>
  `;
}

function create_container_contents(data) {
  console.log(data);
  let contents = "";
  for (var i = data.length - 1; i >= 0; i--) {
    console.log(data[i]);
    contents += create_cards(
      data[i].title,
      data[i].link,
      data[i].date,
      data[i].description
    );
  }
  return contents;
}

function create_cards(title, link, date, description) {
  console.log(title, link, date, description);
  return `
  <div class="card">
    <div class="card-body">
      <h3 class="card-title">${title}</h3>
      <p class="card-text">${date}<br>${description}</p>
    </div>
    <div class="card-footer">
      <a target="_blank" href=${link}>
        <img src="https://img.shields.io/badge/Read_on-Medium-337ab7?style=flat" alt="Read on Medium">
      </a>
    </div>
  </div>
  `;
}

function create_literature_review_section_contents(
  header,
  literature_review_data
) {
  let contents = `<h4>${header}</h4><div class="card-columns">`;
  for (var i = 0; i < literature_review_data.length; i++) {
    contents += `
    <div class="card">
      <div class="card-body">
        <h3 class="card-title">${literature_review_data[i].title}</h3>
      </div>
      <div class="card-footer">
        <a target="_blank" href=${literature_review_data[i].link}>
          <img src="https://img.shields.io/badge/Read_on-Medium-337ab7?style=flat" alt="Read on Medium">
        </a>
      </div>
    </div>`; 
  }
  contents += "</div>";
  return contents;
}

let container = document.getElementById("container");
for (var i = 0; i < nav_data.length; i++) {
  container.innerHTML += create_section(
    `line_${i + 1}`,
    nav_data[i],
    create_cards_container(create_container_contents(papers_data[i]))
  );
}

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


function search() {
  var input, filter, cards;
  input = document.getElementById('search_input');
  filter = input.value.toUpperCase();
  cards = document.getElementsByClassName("card");
  for (i = 0; i < cards.length; i++) {
    if (cards[i].textContent.toUpperCase().indexOf(filter) > -1) {
      cards[i].style.display = "";
    } else {
      cards[i].style.display = "none";
    }
  }
  hide_empty_sections();
}

// if call tha card in a section are hidden, hide the section
function hide_empty_sections() {
  var sections = document.getElementsByClassName("section");
  for (i = 0; i < sections.length; i++) {
    var cards = sections[i].getElementsByClassName("card");
    var visible = false;
    for (j = 0; j < cards.length; j++) {
      if (cards[j].style.display != "none") {
        visible = true;
        break;
      }
    }
    if (!visible) {
      sections[i].style.display = "none";
    } else {
      sections[i].style.display = "";
    }
  }
}