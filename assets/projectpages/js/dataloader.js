const LINK_LABELS = {
  paper: 'Paper',
  demo: 'Demo',
  code: 'Code',
  model: 'Model',
  data: 'Data',
};

document.title = title;
document.getElementById('project-title').innerHTML = title;
document.getElementById('footer-project-name').innerHTML = title;
document.getElementById('project-date').innerHTML = project_date;

const project_links = document.getElementById('project-links');

function create_link_element(link, icon, text) {
  const label = LINK_LABELS[text] || text.charAt(0).toUpperCase() + text.slice(1);
  return `<span class="link-block">
    <a href="${link}" target="_blank" rel="noopener"
      class="external-link button is-normal is-rounded is-dark">
      <span class="icon"><i class="${icon}"></i></span>
      <span>${label}</span>
    </a>
  </span>`;
}

for (const key in links) {
  if (links[key]) {
    project_links.innerHTML += create_link_element(links[key], link2icon[key], key);
  }
}

function renderBlock(content) {
  if (content.type === 'text') {
    return `<p>${content.content}</p>`;
  }
  if (content.type === 'image') {
    return `<figure class="image">
      <img src="${content.src}" alt="${content.alt || ''}" loading="lazy">
    </figure>`;
  }
  if (content.type === 'heading') {
    return `<h3 class="title is-4">${content.content}</h3>`;
  }
  if (content.type === 'list') {
    return `<ul>${content.content.map((item) => `<li>${item}</li>`).join('')}</ul>`;
  }
  if (content.type === 'table') {
    return `<div class="table-container">
      <table class="table is-fullwidth is-bordered is-striped">
        <thead><tr>${content.columns.map((c) => `<th>${c}</th>`).join('')}</tr></thead>
        <tbody>${content.rows.map((row) => `<tr>${row.map((cell) => `<td>${cell}</td>`).join('')}</tr>`).join('')}</tbody>
      </table>
    </div>`;
  }
  if (content.type === 'carousel') {
    return `<div class="carousel results-carousel">
      ${content.images
        .map(
          (image) => `<div class="item">
            <img src="${image.src}" alt="${image.caption || ''}" loading="lazy">
            <h2 class="subtitle has-text-centered">${image.caption || ''}</h2>
          </div>`
        )
        .join('')}
    </div>`;
  }
  if (content.type === 'code') {
    return `<pre class="is-code"><code>${content.content}</code></pre>`;
  }
  if (content.type === 'html') {
    return content.content;
  }
  if (content.type === 'bullet') {
    return `<ul>${content.content.map((item) => `<li>${item}</li>`).join('')}</ul>`;
  }
  return '';
}

function create_content_section(header, content_array, is_hero_light) {
  const hasTable = content_array.some((c) => c.type === 'table' || (c.type === 'html' && c.content.includes('<table')));
  const hasCarousel = content_array.some((c) => c.type === 'carousel');
  const contentClass =
    hasTable || hasCarousel ? 'content' : 'content has-text-justified';

  return `<section class="section ${is_hero_light}">
    <div class="container is-max-desktop">
      <div class="columns is-centered">
        <div class="column is-four-fifths">
          <h2 class="title is-3${hasTable || hasCarousel ? '' : ' has-text-centered'}">${header}</h2>
          <div class="${contentClass}">
            ${content_array.map(renderBlock).join('')}
          </div>
        </div>
      </div>
    </div>
  </section>`;
}

const project_contents_div = document.getElementById('project-contents');
let is_hero_light = 'is-light';
for (const key in project_contents) {
  project_contents_div.innerHTML += create_content_section(
    key,
    project_contents[key],
    is_hero_light
  );
  is_hero_light = is_hero_light === 'is-light' ? '' : 'is-light';
}

function initProjectCarousels() {
  if (typeof bulmaCarousel === 'undefined' || !document.querySelector('.carousel')) {
    return;
  }
  bulmaCarousel.attach('.carousel', {
    slidesToScroll: 1,
    slidesToShow: 1,
    loop: true,
    infinite: true,
    autoplay: true,
    autoplaySpeed: 5000,
    breakpoints: [{ changePoint: 99999, slidesToShow: 1, slidesToScroll: 1 }],
  });
}

document.addEventListener('DOMContentLoaded', initProjectCarousels);
