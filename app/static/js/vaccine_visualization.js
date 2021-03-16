
function render_vaccine_visualization(dailyTotalVaccinations, assumedFutureVaccinations) {

  var vaxData = {
    x: dailyTotalVaccinations['date'], 
    y: dailyTotalVaccinations['vaccinations'], 
    type: 'scatter', 
    mode: 'lines',
    line: {
      dash: 'solid'
    },
    name: 'actual'
  };

  // fake temporary data to play around with
  var futureData = {
    x: assumedFutureVaccinations['date'],
    y: assumedFutureVaccinations['vaccinations'],
    type: 'scatter',
    mode: 'lines',
    line: {
      dash: 'dot'
    },
    name: 'assumed'
  };

  // list of all data series/"traces" (according to plotly)
  var data = [vaxData, futureData];

  var layout = { 
    title: 'Total Vaccinations per Million + Assumed Future Rate',
    paper_bgcolor: '#fafaee'
  };
  
  var config = {responsive: true};

  Plotly.newPlot('vaccine_dataviz', data, layout, config);
}
