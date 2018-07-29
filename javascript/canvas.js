var canvas01 = document.getElementById('python');
var context01 = canvas01.getContext('2d');
var al01=0;
var start01=4.72;
var cw01=context01.canvas.width/2;
var ch01=context01.canvas.height/2;
var diff01;

function progressBar01()
{
  diff01=(al01/100)*Math.PI*2;
  context01.clearRect(0,0,400,200);
  context01.beginPath();
  context01.arc(cw01,ch01,50,0,2*Math.PI,false);
  context01.fillStyle='#FFF';
  context01.fill();
  context01.strokeStyle='#e9ecef';
  context01.stroke();
  context01.fillStyle='#000';
  context01.strokeStyle='#2E9DFF';
  context01.textAlign='center';
  context01.lineWidth=15;
  context01.font = '10pt Verdana';
  context01.beginPath();
  context01.arc(cw01,ch01,50,start01,diff01+start01,false);
  context01.stroke();
  // context.fillText(al+'%',cw+2,ch+6);
  if(al01>=90)
  {
    clearTimeout(bar01);
  }
  al01++;
}

var bar01 = setInterval(progressBar01,50);

var canvas02 = document.getElementById('python-scripting');
var context02 = canvas02.getContext('2d');
var al02=0;
var start02=4.72;
var cw02=context02.canvas.width/2;
var ch02=context02.canvas.height/2;
var diff02;

function progressBar02()
{
  diff02=(al02/100)*Math.PI*2;
  context02.clearRect(0,0,400,200);
  context02.beginPath();
  context02.arc(cw02,ch02,50,0,2*Math.PI,false);
  context02.fillStyle='#FFF';
  context02.fill();
  context02.strokeStyle='#e9ecef';
  context02.stroke();
  context02.fillStyle='#000';
  context02.strokeStyle='#2E9DFF';
  context02.textAlign='center';
  context02.lineWidth=15;
  context02.font = '10pt Verdana';
  context02.beginPath();
  context02.arc(cw02,ch02,50,start02,diff02+start02,false);
  context02.stroke();
  // context.fillText(al+'%',cw+2,ch+6);
  if(al02>=85)
  {
    clearTimeout(bar02);
  }
  al02++;
}

var bar02 = setInterval(progressBar02,50);

var canvas03 = document.getElementById('data-analysis');
var context03 = canvas03.getContext('2d');
var al03=0;
var start03=4.72;
var cw03=context03.canvas.width/2;
var ch03=context03.canvas.height/2;
var diff03;

function progressBar03()
{
  diff03=(al03/100)*Math.PI*2;
  context03.clearRect(0,0,400,200);
  context03.beginPath();
  context03.arc(cw03,ch03,50,0,2*Math.PI,false);
  context03.fillStyle='#FFF';
  context03.fill();
  context03.strokeStyle='#e9ecef';
  context03.stroke();
  context03.fillStyle='#000';
  context03.strokeStyle='#2E9DFF';
  context03.textAlign='center';
  context03.lineWidth=15;
  context03.font = '10pt Verdana';
  context03.beginPath();
  context03.arc(cw03,ch03,50,start03,diff03+start03,false);
  context03.stroke();
  // context.fillText(al+'%',cw+2,ch+6);
  if(al03>=75)
  {
    clearTimeout(bar03);
  }
  al03++;
}

var bar03 = setInterval(progressBar03,50);

var canvas04 = document.getElementById('data-visualisation');
var context04 = canvas04.getContext('2d');
var al04=0;
var start04=4.72;
var cw04=context04.canvas.width/2;
var ch04=context04.canvas.height/2;
var diff04;

function progressBar04()
{
  diff04=(al04/100)*Math.PI*2;
  context04.clearRect(0,0,400,200);
  context04.beginPath();
  context04.arc(cw04,ch04,50,0,2*Math.PI,false);
  context04.fillStyle='#FFF';
  context04.fill();
  context04.strokeStyle='#e9ecef';
  context04.stroke();
  context04.fillStyle='#000';
  context04.strokeStyle='#2E9DFF';
  context04.textAlign='center';
  context04.lineWidth=15;
  context04.font = '10pt Verdana';
  context04.beginPath();
  context04.arc(cw04,ch04,50,start04,diff04+start04,false);
  context04.stroke();
  // context.fillText(al+'%',cw+2,ch+6);
  if(al04>=75)
  {
    clearTimeout(bar04);
  }
  al04++;
}

var bar04 = setInterval(progressBar04,50);

var canvas05 = document.getElementById('machine-learning');
var context05 = canvas05.getContext('2d');
var al05=0;
var start05=4.72;
var cw05=context05.canvas.width/2;
var ch05=context05.canvas.height/2;
var diff05;

function progressBar05()
{
  diff05=(al05/100)*Math.PI*2;
  context05.clearRect(0,0,400,200);
  context05.beginPath();
  context05.arc(cw05,ch05,50,0,2*Math.PI,false);
  context05.fillStyle='#FFF';
  context05.fill();
  context05.strokeStyle='#e9ecef';
  context05.stroke();
  context05.fillStyle='#000';
  context05.strokeStyle='#2E9DFF';
  context05.textAlign='center';
  context05.lineWidth=15;
  context05.font = '10pt Verdana';
  context05.beginPath();
  context05.arc(cw05,ch05,50,start05,diff05+start05,false);
  context05.stroke();
  // context.fillText(al+'%',cw+2,ch+6);
  if(al05>=75)
  {
    clearTimeout(bar05);
  }
  al05++;
}

var bar05 = setInterval(progressBar05,50);

var canvas06 = document.getElementById('java');
var context06 = canvas06.getContext('2d');
var al06=0;
var start06=4.72;
var cw06=context06.canvas.width/2;
var ch06=context06.canvas.height/2;
var diff06;

function progressBar06()
{
  diff06=(al06/100)*Math.PI*2;
  context06.clearRect(0,0,400,200);
  context06.beginPath();
  context06.arc(cw06,ch06,50,0,2*Math.PI,false);
  context06.fillStyle='#FFF';
  context06.fill();
  context06.strokeStyle='#e9ecef';
  context06.stroke();
  context06.fillStyle='#000';
  context06.strokeStyle='#2E9DFF';
  context06.textAlign='center';
  context06.lineWidth=15;
  context06.font = '10pt Verdana';
  context06.beginPath();
  context06.arc(cw06,ch06,50,start06,diff06+start06,false);
  context06.stroke();
  // context.fillText(al+'%',cw+2,ch+6);
  if(al06>=80)
  {
    clearTimeout(bar06);
  }
  al06++;
}

var bar06 = setInterval(progressBar06,50);

var canvas07 = document.getElementById('android');
var context07 = canvas07.getContext('2d');
var al07=0;
var start07=4.72;
var cw07=context07.canvas.width/2;
var ch07=context07.canvas.height/2;
var diff07;

function progressBar07()
{
  diff07=(al07/100)*Math.PI*2;
  context07.clearRect(0,0,400,200);
  context07.beginPath();
  context07.arc(cw07,ch07,50,0,2*Math.PI,false);
  context07.fillStyle='#FFF';
  context07.fill();
  context07.strokeStyle='#e9ecef';
  context07.stroke();
  context07.fillStyle='#000';
  context07.strokeStyle='#2E9DFF';
  context07.textAlign='center';
  context07.lineWidth=15;
  context07.font = '10pt Verdana';
  context07.beginPath();
  context07.arc(cw07,ch07,50,start07,diff07+start07,false);
  context07.stroke();
  // context.fillText(al+'%',cw+2,ch+6);
  if(al07>=65)
  {
    clearTimeout(bar07);
  }
  al07++;
}

var bar07 = setInterval(progressBar07,50);

var canvas08 = document.getElementById('c');
var context08 = canvas08.getContext('2d');
var al08=0;
var start08=4.72;
var cw08=context08.canvas.width/2;
var ch08=context08.canvas.height/2;
var diff08;

function progressBar08()
{
  diff08=(al08/100)*Math.PI*2;
  context08.clearRect(0,0,400,200);
  context08.beginPath();
  context08.arc(cw08,ch08,50,0,2*Math.PI,false);
  context08.fillStyle='#FFF';
  context08.fill();
  context08.strokeStyle='#e9ecef';
  context08.stroke();
  context08.fillStyle='#000';
  context08.strokeStyle='#2E9DFF';
  context08.textAlign='center';
  context08.lineWidth=15;
  context08.font = '10pt Verdana';
  context08.beginPath();
  context08.arc(cw08,ch08,50,start08,diff08+start08,false);
  context08.stroke();
  // context.fillText(al+'%',cw+2,ch+6);
  if(al08>=80)
  {
    clearTimeout(bar08);
  }
  al08++;
}

var bar08 = setInterval(progressBar08,50);

var canvas09 = document.getElementById('html');
var context09 = canvas09.getContext('2d');
var al09=0;
var start09=4.72;
var cw09=context09.canvas.width/2;
var ch09=context09.canvas.height/2;
var diff09;

function progressBar09()
{
  diff09=(al09/100)*Math.PI*2;
  context09.clearRect(0,0,400,200);
  context09.beginPath();
  context09.arc(cw09,ch09,50,0,2*Math.PI,false);
  context09.fillStyle='#FFF';
  context09.fill();
  context09.strokeStyle='#e9ecef';
  context09.stroke();
  context09.fillStyle='#000';
  context09.strokeStyle='#2E9DFF';
  context09.textAlign='center';
  context09.lineWidth=15;
  context09.font = '10pt Verdana';
  context09.beginPath();
  context09.arc(cw09,ch09,50,start09,diff09+start09,false);
  context09.stroke();
  // context.fillText(al+'%',cw+2,ch+6);
  if(al09>=90)
  {
    clearTimeout(bar09);
  }
  al09++;
}

var bar09 = setInterval(progressBar09,50);

var canvas10 = document.getElementById('css');
var context10 = canvas10.getContext('2d');
var al10=0;
var start10=4.72;
var cw10=context10.canvas.width/2;
var ch10=context10.canvas.height/2;
var diff10;

function progressBar10()
{
  diff10=(al10/100)*Math.PI*2;
  context10.clearRect(0,0,400,200);
  context10.beginPath();
  context10.arc(cw10,ch10,50,0,2*Math.PI,false);
  context10.fillStyle='#FFF';
  context10.fill();
  context10.strokeStyle='#e9ecef';
  context10.stroke();
  context10.fillStyle='#000';
  context10.strokeStyle='#2E9DFF';
  context10.textAlign='center';
  context10.lineWidth=15;
  context10.font = '10pt Verdana';
  context10.beginPath();
  context10.arc(cw10,ch10,50,start10,diff10+start10,false);
  context10.stroke();
  // context.fillText(al+'%',cw+2,ch+6);
  if(al10>=90)
  {
    clearTimeout(bar10);
  }
  al10++;
}

var bar10 = setInterval(progressBar10,50);

var canvas11 = document.getElementById('bootstrap');
var context11 = canvas11.getContext('2d');
var al11=0;
var start11=4.72;
var cw11=context11.canvas.width/2;
var ch11=context11.canvas.height/2;
var diff11;

function progressBar11()
{
  diff11=(al11/100)*Math.PI*2;
  context11.clearRect(0,0,400,200);
  context11.beginPath();
  context11.arc(cw11,ch11,50,0,2*Math.PI,false);
  context11.fillStyle='#FFF';
  context11.fill();
  context11.strokeStyle='#e9ecef';
  context11.stroke();
  context11.fillStyle='#000';
  context11.strokeStyle='#2E9DFF';
  context11.textAlign='center';
  context11.lineWidth=15;
  context11.font = '10pt Verdana';
  context11.beginPath();
  context11.arc(cw11,ch11,50,start11,diff11+start11,false);
  context11.stroke();
  // context.fillText(al+'%',cw+2,ch+6);
  if(al11>=90)
  {
    clearTimeout(bar11);
  }
  al11++;
}

var bar11 = setInterval(progressBar11,50);

var canvas12 = document.getElementById('javascript');
var context12 = canvas12.getContext('2d');
var al12=0;
var start12=4.72;
var cw12=context12.canvas.width/2;
var ch12=context12.canvas.height/2;
var diff12;

function progressBar12()
{
  diff12=(al12/100)*Math.PI*2;
  context12.clearRect(0,0,400,200);
  context12.beginPath();
  context12.arc(cw12,ch12,50,0,2*Math.PI,false);
  context12.fillStyle='#FFF';
  context12.fill();
  context12.strokeStyle='#e9ecef';
  context12.stroke();
  context12.fillStyle='#000';
  context12.strokeStyle='#2E9DFF';
  context12.textAlign='center';
  context12.lineWidth=15;
  context12.font = '10pt Verdana';
  context12.beginPath();
  context12.arc(cw12,ch12,50,start12,diff12+start12,false);
  context12.stroke();
  // context.fillText(al+'%',cw+2,ch+6);
  if(al12>=90)
  {
    clearTimeout(bar12);
  }
  al12++;
}

var bar12 = setInterval(progressBar12,50);

var canvas13 = document.getElementById('jquery');
var context13 = canvas13.getContext('2d');
var al13=0;
var start13=4.72;
var cw13=context13.canvas.width/2;
var ch13=context13.canvas.height/2;
var diff13;

function progressBar13()
{
  diff13=(al13/100)*Math.PI*2;
  context13.clearRect(0,0,400,200);
  context13.beginPath();
  context13.arc(cw13,ch13,50,0,2*Math.PI,false);
  context13.fillStyle='#FFF';
  context13.fill();
  context13.strokeStyle='#e9ecef';
  context13.stroke();
  context13.fillStyle='#000';
  context13.strokeStyle='#2E9DFF';
  context13.textAlign='center';
  context13.lineWidth=15;
  context13.font = '10pt Verdana';
  context13.beginPath();
  context13.arc(cw13,ch13,50,start13,diff13+start13,false);
  context13.stroke();
  // context.fillText(al+'%',cw+2,ch+6);
  if(al13>=90)
  {
    clearTimeout(bar13);
  }
  al13++;
}

var bar13 = setInterval(progressBar13,50);
