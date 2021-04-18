
function render_influenza_visualization(influenzaData, nationalCases) {
    var fluData ={
      x: influenzaData['date'],
      y: influenzaData['num_ili'],
      type: 'scatter',
      mode: 'lines',
      line: {
        dash: 'solid'
      },
      marker: {
        color: 'green'
      },
      name: 'Number of ILI'
    };
  
    var caseData = {
      x: nationalCases['date'],
      y: nationalCases['cases'],
      type: 'scatter',
      mode: 'lines',
      line: {
        dash: 'solid'
      },
      marker: {
        color: 'purple'
      },
      name: 'COVID-19 Cases'
    };
  
    var data = [fluData, caseData];
  
    var layout = { 
      title: 'Number FluView Influenza-like Illnesses vs Number of National COVID-19 Cases',
      paper_bgcolor: '#fafaee',
    };
    
    var config = {responsive: true};
  
    Plotly.newPlot('influenza_dataviz', data, layout, config);
  }