
function render_vaccine_visualization(dailyTotalVaccinations) {

  var vaxData = {
    x: dailyTotalVaccinations['date'], 
    y: dailyTotalVaccinations['vaccinations'], 
    type: 'scatter', 
    mode: 'lines',
    line: {
      dash: 'solid'
    },
    name: 'daily vaccinations'
  };

  // fake temporary data to play around with
  var futureData = {
    x: ['2021-03-07', '2021-03-08', '2021-03-09', '2021-03-10', '2021-03-11', '2021-03-12', '2021-03-13', '2021-03-14', 
    '2021-03-15', '2021-03-16', '2021-03-17', '2021-03-18', '2021-03-19', '2021-03-20', '2021-03-21', '2021-03-22', '2021-03-23'],
    y: ['0', '10', '20', '30', '40', '50', '60', '70', '80', '90', '100', '110', '120', '130', '140', '150', '160'],
    type: 'scatter',
    mode: 'lines',
    line: {
      dash: 'dot'
    },
    name: 'predicted ??'
  };

  // list of all data series/"traces" (according to plotly)
  var data = [vaxData, futureData];

  var layout = { 
    title: 'Total Vaccinations per Million + Assumed Future Rate'
  };
  
  var config = {responsive: true};

  Plotly.newPlot('vaccine_dataviz', data, layout, config);
}
