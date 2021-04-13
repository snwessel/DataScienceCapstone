
function render_policy_visualization(statePolicyInfo) {

    // TODO: potential dynamic colorscale would be better, though our data never has more than 6 options currently
    const colors = ["rgb(255, 218, 244)", "rgb(238, 182, 219)", "rgb(222, 146, 193)", "rgb(205, 109, 167)", "rgb(188, 70, 140)", "rgb(170, 5, 112)"];

    var data = [];
    let policy;
    let policyDict;
    // creating a trace/data object per unique gathering ID since there is no way to plot categorical values on chloropleth for plotly.js
    var num = Object.keys(statePolicyInfo).length - 1;
    for (var i = 0; i < num; i++) {
        policy = i.toString();
        let policyTitle = statePolicyInfo[policy]["policy_info"][0];
        
        policyDict = {
            type: 'choropleth',
            locationmode: 'USA-states',
            locations: statePolicyInfo[policy]["state_abbrev"],
            z: statePolicyInfo[policy]["policy_id"],
            text: statePolicyInfo[policy]["state"],
            colorscale: [[0, colors[i]], [1, colors[i]]],
            zmin: i,
            zmax: i + 1,
            colorbar: {
                title: policyTitle,
                x: 0.9,
                y: i / (num * 1.0) + 0.2,
                len: 1.1 / (num * 1.0),
                tick0: i,
                dtick: 2
            },
            name: policyTitle
        };

        data.push(policyDict);
    };

    var layout = {
        title: `${statePolicyInfo["policy_name"]} for Each State`,
        showlegend: true,
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

    var config = { responsive: true };

    Plotly.newPlot('policy_dataviz', data, layout, config);
}
