
function render_prediction_visualization(dailyCases, predictedCases) {
  var caseData ={
    x: dailyCases['date'],
    y: dailyCases['cases'],
    type: 'scatter',
    mode: 'lines',
    line: {
      dash: 'solid'
    },
    name: 'actual'
  };

  // fake temporary data to play around with
  var futureData = {
    x: predictedCases["date"],
    y: predictedCases["predictions"],
    type: 'scatter',
    mode: 'lines',
    line: {
      dash: 'dot'
    },
    name: 'predicted'
  };

  // list of all data series/"traces" 
  var data = [caseData, futureData];

  var layout = { 
    title: 'Daily Covid Cases + Predictions',
    paper_bgcolor: '#fafaee'
  };
  
  var config = {responsive: true};

  Plotly.newPlot('prediction_dataviz', data, layout, config);
}