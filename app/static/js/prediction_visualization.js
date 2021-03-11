
function render_prediction_visualization(dailyCases) {
  var caseData ={
    x: dailyCases['date'],
    y: dailyCases['cases'],
    type: 'scatter',
    mode: 'lines',
    line: {
      dash: 'solid'
    },
    name: 'daily cases'
  };

  // fake temporary data to play around with
  var futureData = {
    x: ['2021-03-07', '2021-03-08', '2021-03-09', '2021-03-10', '2021-03-11', '2021-03-12', '2021-03-13', '2021-03-14', 
    '2021-03-15', '2021-03-16', '2021-03-17', '2021-03-18', '2021-03-19', '2021-03-20', '2021-03-21', '2021-03-22', '2021-03-23'],
    y: ['0', '1000', '2000', '3000', '4000', '5000', '6000', '7000', '8000', '9000', '10000', '11000', '12000', '13000', '14000', '15000', '16000'],
    type: 'scatter',
    mode: 'lines',
    line: {
      dash: 'dot'
    },
    name: 'predicted ???'
  };

  // list of all data series/"traces" 
  var data = [caseData, futureData];

  var layout = { 
    title: 'Daily Covid Cases + Predictions'
  };
  
  var config = {responsive: true};

  Plotly.newPlot('prediction_dataviz', data, layout, config);
}