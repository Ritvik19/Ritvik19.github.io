// alert("linked");
$(document).ready(function(){
  $(".block").hide();
  $("#home").show(2500);
});
// alert("linked");
$("#12").click(function(){
  $("#picture, #header, #intro, #links, #dwnld").hide(2500, function(){
      $("#inavigate").hide();
      $("#home").hide();
      $("#education").show();
      $("#edu-content").hide();
      $("#enavigate").show();
      $("#edu-content").show(2500);
  });
});
$("#13").click(function(){
  $("#picture, #header, #intro, #links, #dwnld").hide(2500, function(){
      $("#inavigate").hide();
      $("#home").hide();
      $("#skills").show();
      $("#skill-content").hide();
      $("#snavigate").show();
      $("#skill-content").show(2500);
  });
});
$("#14").click(function(){
  $("#picture, #header, #intro, #links, #dwnld").hide(2500, function(){
      $("#inavigate").hide();
      $("#home").hide();
      $("#portfolio").show();
      $("#port-content").hide();
      $("#pnavigate").show();
      $("#port-content").show(2500);
  });
});
$("#21").click(function(){
  $("#edu-content").hide(2500, function(){
      $("#enavigate").hide();
      $("#education").hide();
      $("#home").show();
      $("#picture, #header, #intro, #links, #dwnld").hide();
      $("#inavigate").show();
      $("#picture, #header, #intro, #links, #dwnld").show(2500);
  });
});
$("#23").click(function(){
  $("#edu-content").hide(2500, function(){
      $("#enavigate").hide();
      $("#education").hide();
      $("#skills").show();
      $("#skill-content").hide();
      $("#snavigate").show();
      $("#skill-content").show(2500);
  });
});
$("#24").click(function(){
  $("#edu-content").hide(2500, function(){
      $("#enavigate").hide();
      $("#education").hide();
      $("#portfolio").show();
      $("#port-content").hide();
      $("#pnavigate").show();
      $("#port-content").show(2500);
  });
});
$("#31").click(function(){
  $("#skill-content").hide(2500, function(){
      $("#snavigate").hide();
      $("#skills").hide();
      $("#home").show();
      $("#picture, #header, #intro, #links, #dwnld").hide();
      $("#inavigate").show();
      $("#picture, #header, #intro, #links, #dwnld").show(2500);
  });
});
$("#32").click(function(){
  $("#skill-content").hide(2500, function(){
      $("#snavigate").hide();
      $("#skills").hide();
      $("#education").show();
      $("#edu-content").hide();
      $("#enavigate").show();
      $("#edu-content").show(2500);
  });
});
$("#34").click(function(){
  $("#skill-content").hide(2500, function(){
      $("#snavigate").hide();
      $("#skills").hide();
      $("#portfolio").show();
      $("#port-content").hide();
      $("#pnavigate").show();
      $("#port-content").show(2500);
  });
});
$("#41").click(function(){
  $("#port-content").hide(2500, function(){
      $("#pnavigate").hide();
      $("#portfolio").hide();
      $("#home").show();
      $("#picture, #header, #intro, #links, #dwnld").hide();
      $("#inavigate").show();
      $("#picture, #header, #intro, #links, #dwnld").show(2500);
  });
});
$("#42").click(function(){
  $("#port-content").hide(2500, function(){
      $("#pnavigate").hide();
      $("#portfolio").hide();
      $("#education").show();
      $("#edu-content").hide();
      $("#enavigate").show();
      $("#edu-content").show(2500);
  });
});
$("#43").click(function(){
  $("#port-content").hide(2500, function(){
      $("#pnavigate").hide();
      $("#portfolio").hide();
      $("#skills").show();
      $("#skill-content").hide();
      $("#snavigate").show();
      $("#skill-content").show(2500);
  });
});
