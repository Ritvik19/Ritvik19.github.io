document.getElementsByTagName("title")[0].innerHTML =  title;
document.getElementById("project-title").innerHTML = title;
document.getElementById("footer-project-name").innerHTML = title;
document.getElementById("project-date").innerHTML = project_date;
project_links = document.getElementById("project-links");


function create_link_element(link, icon, text){
    return `<span class="link-block">
        <a href="${link}" target="_blank"
            class="external-link button is-normal is-rounded is-dark">
            <span class="icon"><i class="${icon}"></i></span>
            <span>${text}</span>
        </a>
    </span>`;
}
for (let key in links){
    if (links[key] != ""){
        project_links.innerHTML += create_link_element(links[key], link2icon[key], key);
    }
}


function create_content_section(header, content_array, is_hero_light){
    return `<section class="section ${is_hero_light}">
        <div class="container is-max-desktop">
            <div class="columns is-centered has-text-centered">
                <div class="column is-four-fifths">
                    <h2 class="title is-3">${header}</h2>
                    <div class="content has-text-justified">
                        ${content_array.map((content) => {
                            if (content.type == "text"){
                                return `<p>${content.content}</p>`;
                            }
                            if (content.type == "image"){
                                return `<figure class="image">
                                    <img src="${content.src}" alt="${content.alt}">
                                </figure>`;
                            }
                            if (content.type == "heading"){
                                return `<h3 class="title is-4">${content.content}</h3>`;
                            }
                            if (content.type == "list"){
                                return `<ul>${content.content.map((item) => `<li>${item}</li>`).join("")}</ul>`;
                            }
                            if (content.type == "table"){
                                return `<table class="table is-fullwidth">
                                    <thead>
                                        <tr>
                                            ${content.columns.map((column) => `<th>${column}</th>`).join("")}
                                        </tr>
                                    </thead>
                                    <tbody>
                                        ${content.rows.map((row) => `<tr>${row.map((cell) => `<td>${cell}</td>`).join("")}</tr>`).join("")}
                                    </tbody>
                                </table>`;
                            }
                            if (content.type == "carousel"){
                                return `<div class="carousel results-carousel">
                                    ${content.images.map((image) => `<div class="item">
                                        <img src="${image.src}" alt="${image.caption}">
                                        <h2 class="subtitle has-text-centered">${image.caption}</h2>
                                    </div>`).join("")}
                                </div>`;
                            }
                            if (content.type == "code"){
                                return `<pre class="is-code"><code>${content.content}</code></pre>`;
                            }
                            if (content.type == "bullet"){
                                return `<div class="content">
                                    <ul>${content.content.map((item) => `<li>${item}</li>`).join("")}</ul>
                                </div>`;
                            }
                        }).join("")}
                    </div>
                </div>
            </div>
        </div>`
}
    

project_contents_div = document.getElementById("project-contents");

let is_hero_light = "is-dark";
for (let key in project_contents){
    project_contents_div.innerHTML += create_content_section(key, project_contents[key], is_hero_light);
    is_hero_light = is_hero_light == "is-light" ? "is-dark" : "is-light";
}
