function get_nav_item(index, title) {
    return `<li><a href="#line${index}">${title}</a></li>`;
}

function get_container_block(index, title, description) {
    return `
    <section id="line${index}" class="section">
        <div class="row">
            <div class="col-md-12 left-align">
                <h2 class="dark-text">
                    ${title}
                    <hr/>
                </h2>
            </div>
        </div>

        <div class="row">
            <div class="col-md-12">
                <p>${description}</p>
                <p><a href="/readthedocs/${title}" role="button">View Documentation</a></p>
            </div>
        </div>
    </section>`;
}

let nav = document.getElementById("nav");
let container = document.getElementById("container");

for (let i = 0; i < data.length; i++) {
    nav.innerHTML += get_nav_item(i + 1, data[i]["title"]);
    container.innerHTML += get_container_block(
        i + 1,
        data[i]["title"],
        data[i]["description"]
    );
}