
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
        color: 'blue'
      },
      name: 'COVID-19 Cases'
    };
  
    var data = [fluData, caseData];

    // Create rectangular shape highlight to highlight peak flu season
    // Reference: https://plotly.com/javascript/shapes/#highlighting-time-series-regions-with-rectangle-shapes
    var shapes = [
      // 1st highlight from start of graph Jan. 2020 to end of Feb. 2020
      {
        type: 'rect',
        // x-reference is assigned to the x-values
        xref: 'x',
        // y-reference is assigned to the plot paper [0,1]
        yref: 'paper',
        x0: '2020-01-01',
        y0: 0,
        x1: '2020-03-01',
        y1: 1,
        fillcolor: '#FF7276',
        opacity: 0.2,
        line: {
            width: 0
        }
      },
      // 2nd highlight from Dec. 2020 to end of Feb. 2021
      {
        type: 'rect',
        xref: 'x',
        yref: 'paper',
        x0: '2020-12-01',
        y0: 0,
        x1: '2021-03-01',
        y1: 1,
        fillcolor: '#FF7276',
        opacity: 0.2,
        line: {
            width: 0
        }
      }
    ];
    
    var layout = { 
      title: 'Number FluView Influenza-like Illnesses vs Number of National COVID-19 Cases',
      paper_bgcolor: '#fafaee',
      shapes: shapes
    };

    var config = {responsive: true};
  
    Plotly.newPlot('influenza_dataviz', data, layout, config);
  }