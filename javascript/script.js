// alert("linked");
$(document).ready(function(){
  $(".block").hide();
  $("#home").show(5000);
});
// alert("linked");
$("#12").click(function(){
  $("#picture, #header, #intro, #links, #dwnld").slideUp(5000, function(){
      $("#inavigate").hide();
      $("#home").hide();
      $("#skills").show();
      $("#skill-content").hide();
      $("#snavigate").show();
      $("#skill-content").show(5000);
  });
});
