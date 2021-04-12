
function render_prediction_visualization(dailyCases, predictedCases, bounds) {

  var upper_bound ={
    x: predictedCases["date"],
    y: bounds['upper'],
    fill: "tonexty", 
    fillcolor: "rgba(231, 111, 81,0.2)", 
    line: {color: "transparent"}, 
    name: "Upper-bound", 
    showlegend: false, 
    type: "scatter"
  };

  var lower_bound ={
    x: predictedCases["date"],
    y: bounds['lower'],
    fill: "tonexty", 
    fillcolor: "rgba(231, 111, 81,0.2)", 
    line: {color: "transparent"}, 
    name: "Lower-bound", 
    showlegend: false, 
    type: "scatter"
  };

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
  var data = [caseData, futureData, upper_bound, lower_bound];

  var layout = { 
    title: 'Daily Covid Cases + Predictions',
    paper_bgcolor: '#fafaee'
  };
  
  var config = {responsive: true};

  Plotly.newPlot('prediction_dataviz', data, layout, config);
}