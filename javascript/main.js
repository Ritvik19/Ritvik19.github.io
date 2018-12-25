function openNav() {
  document.getElementById("nav").style.width = "100%";
}

function closeNav() {
  document.getElementById("nav").style.width = "0%";
}

$(document).ready(function(){
  $('.disabled').click(function(){
    alert('Sorry for the inconvenience, this project is not available on github')
    return false;
  });
});
