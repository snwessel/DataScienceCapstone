
function render_prediction_visualization(dailyCases) {
  var data = [
    {
      x: dailyCases['date'],
      y: dailyCases['cases'],
      type: 'scatter'
    }
  ];
  
  Plotly.newPlot('prediction_dataviz', data);
}