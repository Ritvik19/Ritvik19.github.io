function createSection(id, title, contents) {
  return `
    <section id="${id}" class="section">
        <div class="row">
            <div class="col-md-12 left-align">
                <h2 class="dark-text">${title}<hr /></h2>
            </div>
        </div>
        <div class="row">
            <div class="col-md-12">${contents}</div>
        </div>
    </section>`;
}

function createCardsContainer(contents) {
  return `<div class="card-columns">${contents}</div>`;
}

function createContainerContents(data) {
  return data.reverse().map(item => createCard(item)).join('');
}

function createTags(tags) {
  if (!tags) return '';
  return `<div class='tags'>${tags.map(tag => `<span class="badge">${tag}</span>`).join('')}</div>`;
}

function createCard({ title, link, date, description, tags }) {
  return `
    <div class="card">
      <div class="card-body">
        <h3 class="card-title">${title}</h3>
        ${date && description? `<p class="card-text">${date}<br>${description} </p>` : ''}
        ${createTags(tags)}
      </div>
      <div class="card-footer">
        <a target="_blank" href=${link}>
          <img src="https://img.shields.io/badge/Read_on-Medium-337ab7?style=flat" alt="Read on Medium">
        </a>
      </div>
    </div>`;
}

function createSurveyCard({title, link, date, description, tags}) {
  return `
    <div class="card">
      <div class="card-body">
        <h3 class="card-title">${title}</h3>
        ${date && description? `<p class="card-text">${date}<br>${description} </p>` : ''}
        ${createTags(tags)}
      </div>
      <div class="card-footer">
        <a target="_blank" href=${link}>
          <img src="https://img.shields.io/badge/Download-pdf-337ab7?style=flat" alt="Download-pdf" download>
        </a>
      </div>
    </div>`;
}

function createSectionWithCards(data) {
  return `
    <div class="card-columns">
      ${data.map(item => createCard(item)).join('')}
    </div>`;
}

function createSurveySection(data) {
  return `
    <div class="card-columns">
      ${data.map(item => createSurveyCard(item)).join('')}
    </div>`;
}

function createLiteratureReviewCard({ title, link, papers }) {
  return `
    <div class="card">
      <div class="card-body">
        <h3 class="card-title">${title}</h3>
      </div>
      <ul class="list-group list-group-flush">
        ${papers.map(paper => `<li class="list-group-item">${paper}</li>`).join('')}
      </ul>
      <div class="card-footer">
        <a target="_blank" href=${link}>
          <img src="https://img.shields.io/badge/Read_on-Medium-337ab7?style=flat" alt="Read on Medium">
        </a>
      </div>
    </div>`;
}

function createLiteratureReviewSection(data) {
  return `
    <div class="card-columns">
      ${data.map(item => createLiteratureReviewCard(item)).join('')}
    </div>`;
}

function populateContainer() {
  let container = document.getElementById("container");

  nav_data.forEach((navItem, index) => {
    container.innerHTML += createSection(
      `line_${index + 1}`,
      navItem,
      createCardsContainer(createContainerContents(papers_data[index]))
    );
  });

  container.innerHTML += createSection(
    `line_${nav_data.length + 1}`,
    "Surveys",
    createSurveySection(surveys_data)
  )

  container.innerHTML += createSection(
    `line_${nav_data.length + 2}`,
    "Literature Reviews",
    createLiteratureReviewSection(literature_review_data)
  );

  container.innerHTML += createSection(
    `line_${nav_data.length + 3}`,
    "Reading Lists",
    createSectionWithCards(reading_list_data)
  );
}

function search() {
  const input = document.getElementById('search_input').value.toUpperCase();
  const cards = document.getElementsByClassName("card");

  Array.from(cards).forEach(card => {
    card.style.display = card.textContent.toUpperCase().includes(input) ? "" : "none";
  });

  hideEmptySections();
}

function hideEmptySections() {
  const sections = document.getElementsByClassName("section");

  Array.from(sections).forEach(section => {
    const visible = Array.from(section.getElementsByClassName("card"))
      .some(card => card.style.display !== "none");

    section.style.display = visible ? "" : "none";
  });
}

function populateNav() {
  nav = document.getElementById("nav");
  nav_data.forEach((navItem, index) => {
    nav.innerHTML += `<li><a href="#line_${index + 1}" class="nav-link">${navItem}</a></li>`;
  });
  nav.innerHTML += `<li><a href="#line_${nav_data.length + 1}" class="nav-link">Surveys</a></li>`;
  nav.innerHTML += `<li><a href="#line_${nav_data.length + 2}" class="nav-link">Literature Reviews</a></li>`;
  nav.innerHTML += `<li><a href="#line_${nav_data.length + 3}" class="nav-link">Reading Lists</a></li>`;
}

populateContainer();
populateNav();
