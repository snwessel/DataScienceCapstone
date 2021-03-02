
function render_prediction_visualization(dailyCases) {
  var data = [
    {
      x: dailyCases['date'],
      y: dailyCases['cases'],
      type: 'scatter'
    }
  ];

  var layout = { 
    title: 'Daily Covid Cases'
  };
  
  var config = {responsive: true};

  Plotly.newPlot('prediction_dataviz', data, layout, config);
}