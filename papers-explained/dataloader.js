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
          <a target="_blank" href="https://ritvik19.medium.com/${link}">
          <img src="https://img.shields.io/badge/Read_on-Medium-337ab7?style=flat" alt="Read on Medium">
        </a>
      </div>
    </div>`;
}

function createSurveyCard({title, link, date, description, tags, papers}) {
  return `
    <div class="card">
      <div class="card-body">
        <h3 class="card-title">${title}</h3>
        ${date && description? `<p class="card-text">${date}<br>${description} </p>` : ''}
        ${papers ? createTags(papers) : ''}
        ${tags ? createTags(tags): ''}
      </div>
      <div class="card-footer">
        <a target="_blank" href="/papers-explained/mind-map/?id=${link}">
          <img src="https://img.shields.io/badge/View_as-Mind_Map-337ab7?style=flat" alt="View-Mind_Map">
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
      ${data.reverse().map(item => createSurveyCard(item)).join('')}
    </div>`;
}

function createLiteratureReviewCard({ title, link, papers }) {
  return `
    <div class="card">
      <div class="card-body">
        <h3 class="card-title">${title}</h3>
        ${createTags(papers)}
      </div>
        
      <div class="card-footer">
        <a target="_blank" href="https://ritvik19.medium.com/${link}">
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

  all_classes.forEach((navItem, index) => {
    container.innerHTML += createSection(
      `line_${index + 1}`,
      navItem,
      createCardsContainer(createContainerContents(papers_data[index]))
    );
  });

  container.innerHTML += createSection(
    `line_${all_classes.length + 1}`,
    "Surveys",
    createSurveySection(surveys_data)
  )

  container.innerHTML += createSection(
    `line_${all_classes.length + 2}`,
    "Journeys",
    createSurveySection(journeys_data)
  );

  container.innerHTML += createSection(
    `line_${all_classes.length + 3}`,
    "Literature Reviews",
    createLiteratureReviewSection(literature_review_data)
  );
}

function search() {
  const queryString = window.location.search;
  const urlParams = new URLSearchParams(queryString);
  const search_query = urlParams.get('search');
  if (search_query) {
    document.getElementById('search_input').value = search_query;
  }
  const input = document.getElementById('search_input').value.toUpperCase();
  const cards = document.getElementsByClassName("card");

  Array.from(cards).forEach(card => {
    card.style.display = card.textContent.toUpperCase().includes(input) ? "" : "none";
  });

  hideEmptySections();
}

function filter() {
  const queryString = window.location.search;
  const urlParams = new URLSearchParams(queryString);
  const filter_query = urlParams.get('tags');
  const filters = filter_query ? filter_query.split(',') : [];
  const cards = document.getElementsByClassName("card");

  Array.from(cards).forEach(card => {
    const card_tags = Array.from(card.getElementsByClassName("badge")).map(tag => tag.textContent);
    card.style.display = filters.every(filter => card_tags.includes(filter)) ? "" : "none";
  });

  hideEmptySections();
}

function hideEmptySections() {
  const sections = document.getElementsByClassName("section");

  Array.from(sections).forEach((section, index) => {
    if (index === 0) {
      section.style.display = "";
      return;
    }

    const visible = Array.from(section.getElementsByClassName("card"))
      .some(card => card.style.display !== "none");

    section.style.display = visible ? "" : "none";
  });
}

function populateNav(){
  const nav = document.getElementById("nav");
  count = 1;
  for (const [category, sub_categories] of Object.entries(nav_data)) {
    console.log(category, sub_categories);
    nav_item = `<li><a href="#line_${count}">${category}</a><ul class="nav">`;
    nav_item += sub_categories.map((sub_category, index) => {
      item = `<li><a href="#line_${count}" class="nav-link">${sub_category}</a></li>`
      count++;
      return item;
    }).join('');
    nav_item += `</ul></li>`;
    nav.innerHTML += nav_item;
  }
}

populateContainer();
populateNav();
search();
filter();