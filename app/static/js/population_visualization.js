
function render_population_visualization(demographicInfo, casesAndDeaths) {

    var data = [];
    let demographic;
    let demographicDict;

    var num = Object.keys(demographicInfo).length;
    // starting from 1 because demographic keys go from 1-6
    for (var i = 1; i < num; i++) {
        demographic = i.toString();
        let demographicName = demographicInfo[demographic]["race"][0];
        // replace every fourth space with break <br> to wrap name in legend
        // help from: https://stackoverflow.com/questions/51097042/how-to-find-and-replace-every-nth-character-occurrence-using-regex
        demographicName = demographicName.replace(/((?:[^\s]*\s){3}[^\s]*)\s/g, '$1<br>');
        
        demographicDict = {
            x: demographicInfo[demographic]["state"],
            y: demographicInfo[demographic]["population"],
            name: demographicName,
            type: 'bar',
        };
        
        // making width of bar shorter for when we are only viewing one state
        if (demographicInfo["state_abbrev"] !== "US") {
          demographicDict["width"] = 0.1;
        };

        data.push(demographicDict);
    };

    // dynamically renaming chart if only a state is selected
    var suffix = (demographicInfo["state_abbrev"] !== "US") ? ` vs<br>Total COVID-19 Cases and Deaths for ${demographicInfo[demographic]["state"][0]} (as of ${casesAndDeaths["date"]})` : ""
  
    var layout = { 
      title: `2019 U.S. Population Estimates by Demographic Breakdown${suffix}`,
      paper_bgcolor: '#fafaee',
      barmode: 'stack'
    };
    
    var config = {responsive: true};
  
    Plotly.newPlot('population_dataviz', data, layout, config);

    // adding extra bar plots when looking at an individual state to compare more information
    if (demographicInfo["state_abbrev"] !== "US" && casesAndDeaths !== null) {

      let deathDict = {
        x: ["Total Deaths"],
        y: casesAndDeaths["deaths"],
        name: "Total Deaths",
        type: 'bar',
        width: 0.1
      };

      let caseDict = {
        x: ["Total Cases"],
        y: casesAndDeaths["cases"],
        name: "Total Cases",
        type: 'bar',
        width: 0.1
      };

      let covidData = [deathDict, caseDict];

      Plotly.addTraces('population_dataviz', covidData);
    };

  }