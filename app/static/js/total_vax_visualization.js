
function render_total_vax_visualization(nationalTotalVaccinations) {

    var vaxData = {
      type: 'choropleth',
      locationmode: 'USA-states',
      locations: nationalTotalVaccinations["abbrev"],
      z: nationalTotalVaccinations["vaccinations"],
      text: nationalTotalVaccinations["location"],
      colorscale: 'YlGnBu',
      reversescale: true,
      autocolorscale: false,
      colorbar: {
        title: 'Total Vaccinations<br>per Hundred'
      }
    };
  
    var data = [vaxData];
  
    var layout = { 
      title: `Total Number of Doses Administered <br> per 100 People in the Total Population of Each State (as of ${nationalTotalVaccinations["date"][0]})`,
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
  
    Plotly.newPlot('total_vax_dataviz', data, layout, config);
  }
  