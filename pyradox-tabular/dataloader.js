function get_nav_item(idx, content) {
  return `<li><a href="#line2_${idx}">${content}</a></li>`;
}

function get_section(id, title, contents) {
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

function get_code_block(code) {
  return `<pre class="brush: python;">${code}</pre>`;
}

function get_reference_block(title, url) {
  return `<li><a target="_blank" href="${url}">${title}</a></li>`;
}

function get_references_contents(references) {
  let contents = "";
  for (let x of references) {
    contents += get_reference_block(x[0], x[1]) + "\n";
  }
  return contents;
}
function get_usage_item(content) {
  let item = "";
  for (let x of content) {
    if (x.type == "p") item += `<p>${x.text}</p>`;
    else if (x.type == "code") item += get_code_block(x.text);
  }
  return item;
}
function get_usage_block(idx, title, content) {
  return `
    <div class="row">
        <div class="col-md-12">
            <h4 id="line2_${idx}">${title}</h4>
            ${content}
        </div>
    </div>`;
}
function get_usage(usage) {
  let contents = "";
  for (var u = 0; u < usage.length; u++) {
    contents += get_usage_block(
      u + 1,
      usage[u].title,
      get_usage_item(usage[u].content)
    );
  }
  return contents;
}

for (var i = 0; i < nav_data.length; i++) {
  document.getElementById("nav").innerHTML += get_nav_item(i + 1, nav_data[i]);
}

let container = document.getElementById("container");
container.innerHTML += get_section(
  "line1",
  "Installation",
  get_code_block("pip install pyradox-tabular")
);
container.innerHTML += get_section("line2", "Usage", get_usage(usage));
container.innerHTML += get_section(
  "line3",
  "References",
  `<ol class="reference">${get_references_contents(references)}</ol>`
);
