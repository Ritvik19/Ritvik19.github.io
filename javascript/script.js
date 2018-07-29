// alert("linked");
$(document).ready(function(){
  $(".block").hide();
  $("#home").show(5000);
});
// alert("linked");
$("#13").click(function(){
  $("#picture, #header, #intro, #links, #dwnld").hide(5000, function(){
      $("#inavigate").hide();
      $("#home").hide();
      $("#skills").show();
      $("#skill-content").hide();
      $("#snavigate").show();
      $("#skill-content").show(5000);
  });
});
$("#31").click(function(){
  $("#skill-content").hide(5000, function(){
      $("#snavigate").hide();
      $("#skills").hide();
      $("#home").show();
      $("#picture, #header, #intro, #links, #dwnld").hide();
      $("#inavigate").show();
      $("#picture, #header, #intro, #links, #dwnld").show(5000);
  });
});
