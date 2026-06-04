function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

function renderActionLinks(actions) {
  return actions
    .map(
      (a) =>
        `<a class="btn btn-ghost" href="${a.link}" target="_blank" rel="noopener">${escapeHtml(a.title)}</a>`
    )
    .join('');
}

function renderResearchCard(item) {
  return `
    <article class="card">
      <h3 class="card-title">${escapeHtml(item.title)}</h3>
      <div class="card-body">${item.description}</div>
      <div class="card-footer">${renderActionLinks(item.actions)}</div>
    </article>`;
}

function renderPublication(item) {
  return `
    <article class="publication-item">
      <h3>${escapeHtml(item.title)}</h3>
      <div class="card-body">${item.description}</div>
      <div class="publication-links">${renderActionLinks(item.actions)}</div>
    </article>`;
}

function renderProjects() {
  const list = document.getElementById('container-projects');
  if (!list || typeof data === 'undefined') return;
  list.innerHTML = `<div class="publication-list">${data.map(renderPublication).join('')}</div>`;
}

function renderResearch() {
  const grid = document.getElementById('container-models');
  if (!grid || typeof models === 'undefined') return;
  grid.innerHTML = `<div class="card-grid">${models.map(renderResearchCard).join('')}</div>`;
}

renderProjects();
renderResearch();
