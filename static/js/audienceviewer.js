const ws = new WebSocket("ws://localhost:8000/publish");
var left_range = [0, 1];

var data0 = [{
    label: "camera0",
    values: [],
    range: left_range
}];
var data1 = [{
    label: "camera1",
    values: [],
    range: left_range
}];
var data2 = [{
    label: "camera2",
    values: [],
    range: left_range
}];
var data3 = [{
    label: "camera3",
    values: [],
    range: left_range
}];

var lineChart0 = $('#graph0').epoch({
    type: 'time.line',
    data: data0,
    axes: ['left', 'right', 'bottom']
});
var lineChart1 = $('#graph1').epoch({
    type: 'time.line',
    data: data1,
    axes: ['left', 'right', 'bottom']
});
var lineChart2 = $('#graph2').epoch({
    type: 'time.line',
    data: data2,
    axes: ['left', 'right', 'bottom']
});
var lineChart3 = $('#graph3').epoch({
    type: 'time.line',
    data: data3,
    axes: ['left', 'right', 'bottom']
});

var graph0 = document.getElementById('graph0');
var graph1 = document.getElementById('graph1');
var graph2 = document.getElementById('graph2');
var graph3 = document.getElementById('graph3');
ws.onmessage = function(msg) {
    console.log(msg);
    var current = JSON.parse(msg.data);
    lineChart0.push([current[0]]);
    lineChart1.push([current[1]]);
    lineChart2.push([current[2]]);
    lineChart3.push([current[3]]);
    var length = Object.keys(current).length;

    /*
    for (var key in current) {
      (function(local) {
        setTimeout(function(){
          console.log(local + ":" + current[local].y);
          $('#graph' + local).css('opacity',current[local].y);
          if (local != 0) {
            for (var i = 0; i < local; i++) {
              $('#graph' + i).css('opacity', 0);
            }
            if (local != length -1) {
              for (var i = local + 1; i < Object.keys(current).length; i++) {
                $('#graph' + i).css('opacity', 0);
              }
            }
          }
          else {
            for (var i = 1; i < length; i++) {
              $('#graph' + i).css('opacity', 0);
            }
          }
          console.log(graph0.style.opacity);
          console.log(graph1.style.opacity);
          console.log(graph2.style.opacity);
          console.log(graph3.style.opacity);
        }, 3000);
      }(key));
    }
    */

    var count = 0;

    var timer = setInterval(function(){
      if (count >= length) {
        clearInterval(timer);
      }
      else {
        console.log(count + ":" + current[count].y);
        $('#graph' + count).css('opacity',current[count].y);
        if (count != 0) {
          for (var i = 0; i < count; i++) {
            $('#graph' + i).css('opacity', 0);
          }
          if (count != length -1) {
            for (var i = count + 1; i < Object.keys(current).length; i++) {
              $('#graph' + i).css('opacity', 0);
            }
          }
        }
        else {
          for (var i = 1; i < length; i++) {
            $('#graph' + i).css('opacity', 0);
          }
        }
        console.log(graph0.style.opacity);
        console.log(graph1.style.opacity);
        console.log(graph2.style.opacity);
        console.log(graph3.style.opacity);
      }
      count++;
    }, 1000);

};
