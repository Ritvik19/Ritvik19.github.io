(function () {
  function navItem(href, label) {
    return `<li><a href="#${href}">${label}</a></li>`;
  }

  function section(id, title, contents) {
    return `
      <section id="${id}" class="docs-section section">
        <h2>${title}</h2>
        ${contents}
      </section>`;
  }

  function codeBlock(code) {
    return `<pre class="brush: python;">${code}</pre>`;
  }

  function tableWrap(inner) {
    return `<div class="table-wrap"><table class="data-table">${inner}</table></div>`;
  }

  function renderNotebookDocs() {
    const nav = document.getElementById('nav');
    const container = document.getElementById('container');
    if (!nav || !container) return;

    for (let i = 0; i < notebook_categories.length; i++) {
      nav.innerHTML += navItem(`line_${i + 1}`, notebook_categories[i]);
      const rows = notebooks_data[i]
        .map(
          (row) =>
            `<tr><td><a href="${row.Link}" target="_blank" rel="noopener">${row.Title}</a></td></tr>`
        )
        .join('');
      container.innerHTML += section(
        `line_${i + 1}`,
        notebook_categories[i],
        tableWrap(`<tbody>${rows}</tbody>`)
      );
    }
  }

  function renderModuleTableDocs(installCmd) {
    const nav = document.getElementById('nav');
    const container = document.getElementById('container');
    if (!nav || !container) return;

    let line = 1;
    if (installCmd) {
      nav.innerHTML += navItem(`line_${line}`, 'Installation');
      container.innerHTML += section(`line_${line}`, 'Installation', codeBlock(installCmd));
      line++;
    }

    for (let i = 0; i < nav_data.length; i++) {
      nav.innerHTML += navItem(`line_${line + i}`, nav_data[i]);
      const rows = usage_data[i]
        .map(
          (row) => `<tr>
            <td><a href="${row.Usage}" target="_blank" rel="noopener">${row.Module}</a></td>
            <td>${row.Description}</td>
            <td>${row['Input Shape']}</td>
            <td>${row['Output Shape']}</td>
          </tr>`
        )
        .join('');
      container.innerHTML += section(
        `line_${line + i}`,
        nav_data[i],
        tableWrap(
          `<thead><tr><th>Module</th><th>Description</th><th>Input Shape</th><th>Output Shape</th></tr></thead><tbody>${rows}</tbody>`
        )
      );
    }
  }

  function usageItemContent(content) {
    return content
      .map((x) => {
        if (x.type === 'p') return `<p>${x.text}</p>`;
        if (x.type === 'code') return codeBlock(x.text);
        return '';
      })
      .join('');
  }

  function renderUsageBlocks(usage, idPrefix) {
    return usage
      .map(
        (u, idx) => `
        <div class="usage-block">
          <h3 id="${idPrefix}_${idx + 1}">${u.title}</h3>
          ${usageItemContent(u.content)}
        </div>`
      )
      .join('');
  }

  function renderUsageReferencesDocs(installCmd) {
    const nav = document.getElementById('nav');
    const container = document.getElementById('container');
    if (!nav || !container) return;

    nav_data.forEach((item, i) => {
      nav.innerHTML += navItem(`line2_${i + 1}`, item);
    });

    container.innerHTML += section('line1', 'Installation', codeBlock(installCmd));
    container.innerHTML += section('line2', 'Usage', renderUsageBlocks(usage, 'line2'));

    if (typeof references !== 'undefined') {
      const refs = references
        .map((ref, i) => {
          const title = Array.isArray(ref) ? ref[0] : ref.title;
          const url = Array.isArray(ref) ? ref[1] : ref.url;
          const idAttr = references[0] && Array.isArray(references[0]) ? ` id="ref-${i + 1}"` : '';
          return `<li><a${idAttr} href="${url}" target="_blank" rel="noopener">${title}</a></li>`;
        })
        .join('');
      container.innerHTML += section('line3', 'References', `<ol class="reference">${refs}</ol>`);
    }
  }

  function renderUsageExamplesDocs(installCmd) {
    const nav = document.getElementById('nav');
    const container = document.getElementById('container');
    if (!nav || !container) return;

    nav_data.forEach((item, i) => {
      nav.innerHTML += navItem(`line2_${i + 1}`, item);
    });

    container.innerHTML += section('line1', 'Installation', codeBlock(installCmd));
    container.innerHTML += section('line2', 'Usage', renderUsageBlocks(usage, 'line2'));

    if (typeof examples !== 'undefined') {
      const ex = examples
        .map(([title, url]) => `<li><a href="${url}" target="_blank" rel="noopener">${title}</a></li>`)
        .join('');
      container.innerHTML += section('line3', 'Examples', `<ul>${ex}</ul>`);
    }
  }

  function initSyntaxHighlighter() {
    if (typeof SyntaxHighlighter !== 'undefined') {
      SyntaxHighlighter.all();
    }
  }

  function detectAndRender() {
    if (typeof notebook_categories !== 'undefined') {
      renderNotebookDocs();
    } else if (typeof usage_data !== 'undefined' && usage_data[0]?.[0]?.Module) {
      const install =
        typeof docsInstallCommand !== 'undefined'
          ? docsInstallCommand
          : 'pip install pyradox-generative';
      renderModuleTableDocs(install);
    } else if (typeof usage !== 'undefined' && typeof examples !== 'undefined') {
      const install = typeof docsInstallCommand !== 'undefined' ? docsInstallCommand : 'pip install vizard';
      renderUsageExamplesDocs(install);
    } else if (typeof usage !== 'undefined' && typeof references !== 'undefined') {
      const install =
        typeof docsInstallCommand !== 'undefined'
          ? docsInstallCommand
          : 'pip install pyradox-generative';
      renderUsageReferencesDocs(install);
    } else if (typeof usage !== 'undefined') {
      const install =
        typeof docsInstallCommand !== 'undefined'
          ? docsInstallCommand
          : 'pip install pyradox-generative';
      renderUsageReferencesDocs(install);
    }
    initSyntaxHighlighter();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', detectAndRender);
  } else {
    detectAndRender();
  }
})();
