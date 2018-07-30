$(document).ready(function(){
  $(".block").hide();
  $("#home").show(2500);
  $(".back").fadeOut();
});


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


$("#p01f").mouseenter(function(){
  $("#p01f").fadeOut(1250, function(){
      $("#p01b").fadeIn(1250);
  });
});
$("#p01b").mouseleave(function(){
  $("#p01b").fadeOut(1250, function(){
    $("#p01f").fadeIn(1250);
  });
});

$("#p02f").mouseenter(function(){
  $("#p02f").fadeOut(1250, function(){
      $("#p02b").fadeIn(1250);
  });
});
$("#p02b").mouseleave(function(){
  $("#p02b").fadeOut(1250, function(){
    $("#p02f").fadeIn(1250);
  });
});

$("#p03f").mouseenter(function(){
  $("#p03f").fadeOut(1250, function(){
      $("#p03b").fadeIn(1250);
  });
});
$("#p03b").mouseleave(function(){
  $("#p03b").fadeOut(1250, function(){
    $("#p03f").fadeIn(1250);
  });
});

$("#p04f").mouseenter(function(){
  $("#p04f").fadeOut(1250, function(){
      $("#p04b").fadeIn(1250);
  });
});
$("#p04b").mouseleave(function(){
  $("#p04b").fadeOut(1250, function(){
    $("#p04f").fadeIn(1250);
  });
});

$("#p05f").mouseenter(function(){
  $("#p05f").fadeOut(1250, function(){
      $("#p05b").fadeIn(1250);
  });
});
$("#p05b").mouseleave(function(){
  $("#p05b").fadeOut(1250, function(){
    $("#p05f").fadeIn(1250);
  });
});

$("#p06f").mouseenter(function(){
  $("#p06f").fadeOut(1250, function(){
      $("#p06b").fadeIn(1250);
  });
});
$("#p06b").mouseleave(function(){
  $("#p06b").fadeOut(1250, function(){
    $("#p06f").fadeIn(1250);
  });
});

$("#p07f").mouseenter(function(){
  $("#p07f").fadeOut(1250, function(){
      $("#p07b").fadeIn(1250);
  });
});
$("#p07b").mouseleave(function(){
  $("#p07b").fadeOut(1250, function(){
    $("#p07f").fadeIn(1250);
  });
});

$("#p08f").mouseenter(function(){
  $("#p08f").fadeOut(1250, function(){
      $("#p08b").fadeIn(1250);
  });
});
$("#p08b").mouseleave(function(){
  $("#p08b").fadeOut(1250, function(){
    $("#p08f").fadeIn(1250);
  });
});
