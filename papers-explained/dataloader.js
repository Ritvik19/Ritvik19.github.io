function createSection(id, title, contents) {
  return `
    <section id="${id}" class="docs-section section">
      <h2>${title}</h2>
      ${contents}
    </section>`;
}

function createTags(tags) {
  if (!tags) return '';
  return `<div class="tags">${tags.map((tag) => `<span class="badge">${tag}</span>`).join('')}</div>`;
}

function createCard({ title, link, date, description, tags }) {
  return `
    <article class="paper-card card">
      <h3>${title}</h3>
      ${date || description ? `<p class="card-text">${date ? `${date}<br>` : ''}${description || ''}</p>` : ''}
      ${createTags(tags)}
      <div class="card-footer">
        <a target="_blank" rel="noopener" href="https://ritvik19.medium.com/${link}">
          <img src="https://img.shields.io/badge/Read_on-Medium-337ab7?style=flat" alt="Read on Medium">
        </a>
      </div>
    </article>`;
}

function createSurveyCard({ title, link, date, description, tags, papers }) {
  return `
    <article class="paper-card card">
      <h3>${title}</h3>
      ${date || description ? `<p class="card-text">${date ? `${date}<br>` : ''}${description || ''}</p>` : ''}
      ${papers ? createTags(papers) : ''}
      ${tags ? createTags(tags) : ''}
      <div class="card-footer">
        <a href="/papers-explained/mind-map/?id=${link}">
          <img src="https://img.shields.io/badge/View_as-Mind_Map-337ab7?style=flat" alt="View Mind Map">
        </a>
      </div>
    </article>`;
}

function createLiteratureReviewCard({ title, link, papers }) {
  return `
    <article class="paper-card card">
      <h3>${title}</h3>
      ${createTags(papers)}
      <div class="card-footer">
        <a target="_blank" rel="noopener" href="https://ritvik19.medium.com/${link}">
          <img src="https://img.shields.io/badge/Read_on-Medium-337ab7?style=flat" alt="Read on Medium">
        </a>
      </div>
    </article>`;
}

function createPaperGrid(cardsHtml) {
  return `<div class="paper-grid">${cardsHtml}</div>`;
}

function createContainerContents(data) {
  return data
    .slice()
    .reverse()
    .map((item) => createCard(item))
    .join('');
}

function populateContainer() {
  const container = document.getElementById('container');

  all_classes.forEach((navItem, index) => {
    container.innerHTML += createSection(
      `line_${index + 1}`,
      navItem,
      createPaperGrid(createContainerContents(papers_data[index]))
    );
  });

  container.innerHTML += createSection(
    `line_${all_classes.length + 1}`,
    'Surveys',
    createPaperGrid(surveys_data.slice().reverse().map((item) => createSurveyCard(item)).join(''))
  );

  container.innerHTML += createSection(
    `line_${all_classes.length + 2}`,
    'Journeys',
    createPaperGrid(journeys_data.slice().reverse().map((item) => createSurveyCard(item)).join(''))
  );

  container.innerHTML += createSection(
    `line_${all_classes.length + 3}`,
    'Literature Reviews',
    createPaperGrid(literature_review_data.map((item) => createLiteratureReviewCard(item)).join(''))
  );
}

function search() {
  const queryString = window.location.search;
  const urlParams = new URLSearchParams(queryString);
  const search_query = urlParams.get('search');
  if (search_query) {
    document.getElementById('search_input').value = search_query;
  }
  const input = document.getElementById('search_input').value.toUpperCase().replace(/[ -]/g, '');
  const cards = document.getElementsByClassName('paper-card');

  Array.from(cards).forEach((card) => {
    const h3 = card.getElementsByTagName('h3')[0];
    const match =
      card.textContent.toUpperCase().includes(input) ||
      (h3 && h3.textContent.toUpperCase().replace(/[ -]/g, '').includes(input));
    card.style.display = match ? '' : 'none';
  });

  hideEmptySections();
  scheduleMasonryLayout();
}

function filter() {
  const queryString = window.location.search;
  const urlParams = new URLSearchParams(queryString);
  const filter_query = urlParams.get('tags');
  const filters = filter_query ? filter_query.split(',') : [];
  const cards = document.getElementsByClassName('paper-card');

  Array.from(cards).forEach((card) => {
    const card_tags = Array.from(card.getElementsByClassName('badge')).map((tag) => tag.textContent);
    card.style.display = filters.every((f) => card_tags.includes(f)) ? '' : 'none';
  });

  hideEmptySections();
  scheduleMasonryLayout();
}

function hideEmptySections() {
  const sections = document.querySelectorAll('.docs-section');

  Array.from(sections).forEach((section) => {
    const visible = Array.from(section.getElementsByClassName('paper-card')).some(
      (card) => card.style.display !== 'none'
    );
    section.style.display = visible ? '' : 'none';
  });
}

function getMasonryColumnCount() {
  if (window.matchMedia('(max-width: 600px)').matches) return 1;
  if (window.matchMedia('(max-width: 900px)').matches) return 2;
  return 3;
}

function getPaperGridGap() {
  const root = getComputedStyle(document.documentElement);
  const fontSize = parseFloat(root.fontSize) || 16;
  const spaceMd = root.getPropertyValue('--space-md').trim();
  if (spaceMd.endsWith('rem')) return parseFloat(spaceMd) * fontSize;
  if (spaceMd.endsWith('px')) return parseFloat(spaceMd);
  return 16;
}

function isCardVisible(card) {
  return card.style.display !== 'none' && getComputedStyle(card).display !== 'none';
}

function resetMasonryCard(card) {
  card.style.position = '';
  card.style.left = '';
  card.style.top = '';
  card.style.width = '';
  card.style.visibility = '';
}

function layoutMasonryGrids() {
  const gap = getPaperGridGap();

  document.querySelectorAll('.container--papers .paper-grid').forEach((grid) => {
    const cols = getMasonryColumnCount();
    const cards = [...grid.querySelectorAll('.paper-card')];

    cards.forEach(resetMasonryCard);
    grid.style.height = '';

    const visibleCards = cards.filter(isCardVisible);

    if (cols === 1) {
      return;
    }

    const gridWidth = grid.clientWidth;
    if (!gridWidth || !visibleCards.length) {
      grid.style.height = '0';
      return;
    }

    const colWidth = (gridWidth - gap * (cols - 1)) / cols;
    const colHeights = new Array(cols).fill(0);

    visibleCards.forEach((card) => {
      card.style.position = 'absolute';
      card.style.width = `${colWidth}px`;
      card.style.visibility = 'hidden';
      card.style.left = '0';
      card.style.top = '0';
    });

    visibleCards.forEach((card) => {
      const col = colHeights.indexOf(Math.min(...colHeights));
      const left = col * (colWidth + gap);
      const top = colHeights[col];
      const height = card.offsetHeight;

      card.style.visibility = '';
      card.style.left = `${left}px`;
      card.style.top = `${top}px`;
      colHeights[col] += height + gap;
    });

    const maxHeight = Math.max(...colHeights, 0);
    grid.style.height = maxHeight > 0 ? `${maxHeight - gap}px` : '0';
  });
}

let masonryFrame;

function scheduleMasonryLayout() {
  cancelAnimationFrame(masonryFrame);
  masonryFrame = requestAnimationFrame(layoutMasonryGrids);
}

function setupMasonry() {
  scheduleMasonryLayout();

  if (window.__papersMasonryReady) return;
  window.__papersMasonryReady = true;

  window.addEventListener('resize', scheduleMasonryLayout);

  window.__papersMasonryResizeObserver = new ResizeObserver(scheduleMasonryLayout);
  document.querySelectorAll('.container--papers .paper-grid, .container--papers .paper-card').forEach((el) => {
    window.__papersMasonryResizeObserver.observe(el);
  });
}

function populateNav() {
  const nav = document.getElementById('nav');
  let count = 1;
  for (const [category, sub_categories] of Object.entries(nav_data)) {
    let nav_item = `<li><a href="#line_${count}">${category}</a><ul class="nav">`;
    nav_item += sub_categories
      .map((sub_category) => {
        const item = `<li><a href="#line_${count}">${sub_category}</a></li>`;
        count++;
        return item;
      })
      .join('');
    nav_item += '</ul></li>';
    nav.innerHTML += nav_item;
  }
}

populateContainer();
populateNav();
search();
filter();
setupMasonry();

if (document.fonts && document.fonts.ready) {
  document.fonts.ready.then(scheduleMasonryLayout);
}

window.addEventListener('load', scheduleMasonryLayout);
