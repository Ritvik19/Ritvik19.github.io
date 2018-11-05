$('#nav').hide();

$('#show').click(function(){
  $('#nav').show(1000);
  $('#main').hide(1000);
  // $('#nav')[0].style.width = "80vw";
});

$('#hide').click(function(){
  $('#nav').hide(1000);
  $('#main').show(1000);
  // $('#nav')[0].style.width = "0vw";
});
