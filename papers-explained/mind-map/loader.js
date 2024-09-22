const queryString = window.location.search;
const urlParams = new URLSearchParams(queryString);
const paperId = urlParams.get('id');

console.log(paperId);

document.getElementById("paper_id").src = `../data/${paperId}.js`;
document.title = `Papers Explained - ${paperId}`;
document.getElementById("title").innerHTML = paperId;