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

let container = document.getElementById("container");
container.innerHTML += get_section(
    "line1",
    "Machine Learning",
    get_table(get_contents(data["ML"]))
);
container.innerHTML += get_section(
    "line2",
    "Deep Learning",
    get_table(get_contents(data["DL"]))
);
container.innerHTML += get_section(
    "line3",
    "Natural Language Processing",
    get_table(get_contents(data["NLP"]))
);
container.innerHTML += get_section(
    "line4",
    "Computer Vision",
    get_table(get_contents(data["CV"]))
);