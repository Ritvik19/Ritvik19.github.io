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

function get_table(contents) {
  return `
    <table class="table">
        <tbody>
            ${contents}
        </tbody>
    </table>
    `;
}

function get_contents(data) {
  let contents = "";
  for (let x of data) {
    contents += get_row(x) + "\n";
  }
  return contents;
}

function get_row(contents) {
  return `
    <tr>
        <td>${contents[0]}</td>
        <td><a href="${contents[1]}", target="_blank">GitHub</a></td>
        <td><a href="${contents[2]}", target="_blank">nbviewer</a></td>
        <td><a href="${contents[3]}", target="_blank">Kaggle</a></td>
    </tr>
    `;
}

function get_nav_li(id, title) {
  return `<li><a href="#${id}">${title}</a></li>`;
}

let container = document.getElementById("container");
let nav_menu = document.getElementById("nav-menu");

let meta_data = [
  ["line01", "ML", "Machine Learning"],
  ["line02", "DL", "Deep Learning"],
  ["line03", "TDL", "Tabular Deep Learning"],
  ["line04", "NLP", "Natural Language Processing"],
  ["line05", "RNN", "Recurrent Neural Networks"],
  ["line06", "TRF", "Transformers"],
  ["line07", "CV", "Computer Vision"],
  ["line08", "CNN", "Convolutional Neural Networks"],
  ["line09", "OD", "Object Detection"],
  ["line10", "AE", "Autoencoders"],
  ["line11", "GAN", "Generative Adversarial Networks"],
  ["line12", "LayoutLM", "Layout LM"],
];

for (let row of meta_data) {
  nav_menu.innerHTML += get_nav_li(row[0], row[2]);
  container.innerHTML += get_section(
    row[0],
    row[2],
    get_table(get_contents(data[row[1]]))
  );
}
