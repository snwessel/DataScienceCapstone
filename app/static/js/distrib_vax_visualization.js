
function render_distrib_vax_visualization(nationalDistribVaccinations) {

    var vaxData = {
      type: 'choropleth',
      locationmode: 'USA-states',
      locations: nationalDistribVaccinations["abbrev"],
      z: nationalDistribVaccinations["vaccinations"],
      text: nationalDistribVaccinations["location"],
      colorscale: 'Reds',
      autocolorscale: false,
      colorbar: {
        title: 'Distributed Vaccinations per Hundred'
      }
    };
  
    var data = [vaxData];
  
    var layout = { 
      title: `Cumulative Counts of COVID-19 Vaccine Doses Recorded as Shipped in CDC's Vaccine Tracking System <br> per 100 People in the Total Population of Each State (as of ${nationalDistribVaccinations["date"][0]})`,
      paper_bgcolor: '#fafaee',
      geo: {
        scope: 'usa',
        countrycolor: 'rgb(255, 255, 255)',
        showland: true,
        landcolor: 'rgb(217, 217, 217)',
        lakecolor: 'rgb(255, 255, 255)',
        subunitcolor: 'rgb(255, 255, 255)',
        lonaxis: {},
        lataxis: {}
        }
    };
    
    var config = {responsive: true};
  
    Plotly.newPlot('distrib_vax_dataviz', data, layout, config);
  }
  