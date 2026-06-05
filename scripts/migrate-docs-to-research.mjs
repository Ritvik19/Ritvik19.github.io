#!/usr/bin/env node
import fs from 'fs';
import path from 'path';
import vm from 'vm';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.join(__dirname, '..');

const LINK2ICON = `let link2icon = {
    "code": "fas fa-code",
    "demo": "fas fa-terminal",
    "model": "fas fa-cogs",
    "data": "fas fa-database",
    "paper": "fas fa-file-pdf",
}`;

function loadDataJs(filePath) {
  const code = fs.readFileSync(filePath, 'utf8');
  const fn = new Function(`
${code}
return {
  usage: typeof usage !== 'undefined' ? usage : undefined,
  references: typeof references !== 'undefined' ? references : undefined,
  nav_data: typeof nav_data !== 'undefined' ? nav_data : undefined,
  usage_data: typeof usage_data !== 'undefined' ? usage_data : undefined,
  notebook_categories: typeof notebook_categories !== 'undefined' ? notebook_categories : undefined,
  notebooks_data: typeof notebooks_data !== 'undefined' ? notebooks_data : undefined,
};
`);
  return fn();
}

function escapeForJs(str) {
  return JSON.stringify(str);
}

function convertUsageBlock(item) {
  if (item.type === 'p') return { type: 'text', content: item.text };
  if (item.type === 'code') return { type: 'code', content: item.text };
  if (item.type === 'html') return { type: 'html', content: item.text };
  return null;
}

function convertUsageToProjectContents(usage, references, installCmd, overviewText) {
  const project_contents = {};
  if (overviewText) {
    project_contents.Overview = [{ type: 'text', content: overviewText }];
  }
  if (installCmd) {
    project_contents.Installation = [{ type: 'code', content: installCmd }];
  }
  for (const section of usage) {
    project_contents[section.title] = section.content
      .map(convertUsageBlock)
      .filter(Boolean);
  }
  if (references?.length) {
    project_contents.References = [
      {
        type: 'list',
        content: references.map(([title, url]) =>
          `<a href="${url}" target="_blank" rel="noopener">${title}</a>`
        ),
      },
    ];
  }
  return project_contents;
}

function convertNotebooks(notebook_categories, notebooks_data, overviewText) {
  const project_contents = {};
  if (overviewText) {
    project_contents.Overview = [{ type: 'text', content: overviewText }];
  }
  for (let i = 0; i < notebook_categories.length; i++) {
    const cat = notebook_categories[i];
    const rows = notebooks_data[i].map((row) => [
      `<a href="${row.Link}" target="_blank" rel="noopener">${row.Title}</a>`,
    ]);
    project_contents[cat] = [
      { type: 'table', columns: ['Notebook'], rows },
    ];
  }
  return project_contents;
}

function convertModuleTables(nav_data, usage_data, installCmd, overviewText) {
  const project_contents = {};
  if (overviewText) {
    project_contents.Overview = [{ type: 'text', content: overviewText }];
  }
  if (installCmd) {
    project_contents.Installation = [{ type: 'code', content: installCmd }];
  }
  for (let i = 0; i < nav_data.length; i++) {
    const rows = usage_data[i].map((row) => [
      `<a href="${row.Usage}" target="_blank" rel="noopener">${row.Module}</a>`,
      row.Description,
      row['Input Shape'],
      row['Output Shape'],
    ]);
    project_contents[nav_data[i]] = [
      {
        type: 'table',
        columns: ['Module', 'Description', 'Input Shape', 'Output Shape'],
        rows,
      },
    ];
  }
  return project_contents;
}

function emitDataJs(meta, project_contents) {
  const linksStr = Object.entries(meta.links)
    .map(([k, v]) => `    "${k}": ${escapeForJs(v)}`)
    .join(',\n');

  let sections = 'let project_contents = {\n';
  for (const [key, blocks] of Object.entries(project_contents)) {
    sections += `    ${escapeForJs(key)}: ${JSON.stringify(blocks, null, 4)
      .split('\n')
      .map((line, i) => (i === 0 ? line : '    ' + line))
      .join('\n')},\n`;
  }
  sections += '};';

  return `let title = ${escapeForJs(meta.title)};
let project_date = ${escapeForJs(meta.project_date)}
let links = {
${linksStr}
}
${LINK2ICON}
${sections}
`;
}

const projects = [
  {
    dir: 'nanoformers',
    type: 'usage',
    meta: {
      title: 'Nanoformers',
      project_date: 'Open Source',
      links: {
        paper: '',
        demo: 'https://wandb.ai/ritvik19/nanoformers',
        code: 'https://github.com/Ritvik19/Nanoformers',
        model: '',
        data: '',
      },
      overview:
        'Miniature implementations of key LLMs — a minimal playground for building and training transformer models from scratch. Covers self-supervised, supervised, reinforcement, and contrastive learning with tiny transformer architectures, PEFT (LoRA/QLoRA), and 20+ training runs logged on Weights &amp; Biases.',
      install:
        'git clone https://github.com/Ritvik19/Nanoformers.git\ncd Nanoformers\npip install -r requirements.txt',
    },
  },
  {
    dir: 'data-science-from-scratch',
    type: 'notebooks',
    meta: {
      title: 'Data Science from Scratch',
      project_date: 'Open Source',
      links: {
        paper: '',
        demo: '',
        code: '',
        model: '',
        data: '',
      },
      overview:
        'Implementations of data science techniques and research papers as Kaggle notebooks. This project garnered attention from Elvis Saravia (ex Cofounder of paperswithcode.com), who added some implementations to the <a href="https://github.com/dair-ai/ML-Notebooks" target="_blank" rel="noopener">ML-Notebooks</a> project.',
    },
  },
  {
    dir: 'pyradox',
    type: 'modules',
    meta: {
      title: 'pyradox',
      project_date: 'Open Source',
      links: {
        paper: '',
        demo: '',
        code: 'https://github.com/Ritvik19/pyradox',
        model: '',
        data: '',
      },
      overview:
        'State-of-the-art neural networks for deep learning with TensorFlow 2. This library helps you implement various state-of-the-art neural networks in a fully customizable fashion. See also <a href="/pyradox-generative/">pyradox-generative</a> and <a href="/pyradox-tabular/">pyradox-tabular</a>.',
      install: 'pip install pyradox',
    },
  },
  {
    dir: 'pyradox-generative',
    type: 'usage',
    meta: {
      title: 'pyradox-generative',
      project_date: 'Open Source',
      links: {
        paper: '',
        demo: '',
        code: 'https://github.com/Ritvik19/pyradox-generative',
        model: '',
        data: '',
      },
      overview:
        'Lightweight trainers for various state-of-the-art Generative Adversarial Networks. Part of the <a href="/pyradox/">pyradox</a> ecosystem.',
      install: 'pip install pyradox-generative',
    },
  },
  {
    dir: 'pyradox-tabular',
    type: 'usage',
    meta: {
      title: 'pyradox-tabular',
      project_date: 'Open Source',
      links: {
        paper: '',
        demo: '',
        code: 'https://github.com/Ritvik19/pyradox-tabular',
        model: '',
        data: '',
      },
      overview:
        'Implementations for various state-of-the-art neural networks for tabular data. Part of the <a href="/pyradox/">pyradox</a> ecosystem.',
      install: 'pip install pyradox-tabular',
    },
  },
];

for (const proj of projects) {
  const dataPath = path.join(root, proj.dir, 'data.js');
  const sandbox = loadDataJs(dataPath);
  let project_contents;

  if (proj.type === 'usage') {
    project_contents = convertUsageToProjectContents(
      sandbox.usage,
      sandbox.references,
      proj.meta.install,
      proj.meta.overview
    );
  } else if (proj.type === 'notebooks') {
    project_contents = convertNotebooks(
      sandbox.notebook_categories,
      sandbox.notebooks_data,
      proj.meta.overview
    );
  } else if (proj.type === 'modules') {
    project_contents = convertModuleTables(
      sandbox.nav_data,
      sandbox.usage_data,
      proj.meta.install,
      proj.meta.overview
    );
  }

  const out = emitDataJs(proj.meta, project_contents);
  fs.writeFileSync(dataPath, out);
  console.log(`Migrated ${proj.dir}/data.js`);
}
