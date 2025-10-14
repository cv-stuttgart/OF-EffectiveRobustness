
// Using Map guarantees iteration in insertion order
const BASELINEFILES = new Map([
    ["Sintel", "./static/data/baseline_FlyingThings3D_Sintel.json"],
    ["KITTI", "./static/data/baseline_FlyingThings3D_Kitti15.json"],
    ["HD1K", "./static/data/baseline_FlyingThings3D_HD1KSplitScheurer.json"],
    ["Driving", "./static/data/baseline_FlyingThings3D_Driving.json"],
    ["VIPER", "./static/data/baseline_FlyingThings3D_Viper.json"],
    ["Spring", "./static/data/baseline_FlyingThings3D_Spring.json"],
]);

const YourModelWauc = {};
const PlotData = [];
let Plot;

function logit(x) {
    return Math.log(x / (1 - x));
}

function expit(x) {
    return 1 / (1 + Math.exp(-x));
}

async function calculateEffectiveRobustness(waucID, waucOOD, baselinefile) {
    try {
        const baselineJSON = await fetch(baselinefile);
        const baselineData = await baselineJSON.json();
        const coeff_a = baselineData["coeff_a"];
        const coeff_b = baselineData["coeff_b"];

        const baselineWAUC = expit(coeff_a * logit(waucID) + coeff_b);
        console.debug("baselineWAUC", baselineWAUC, "coeff_a", coeff_a, "coeff_b", coeff_b, "waucID", waucID, "waucOOD", waucOOD);

        const effectiveRobustness = (waucOOD - baselineWAUC);
        console.debug("effectiveRobustness", effectiveRobustness);
        return effectiveRobustness;
    } catch (error) {
        console.error("Error loading baseline file:", error);
        return null;
    }
}


async function loadCSV(filePath) {
    try {
        // Fetch the CSV file
        const response = await fetch(filePath);
        const csvText = await response.text();
        const rows = csvText.split("\n").map(row => row.split(",")).filter(row => row.length > 1);
        const headers = rows[0].map(header => header.trim());
        const data = rows.slice(1).map(row => {
            const obj = {};
            headers.forEach((header, index) => {
                obj[header] = row[index].trim();
            });
            return obj;
        });
        return { data, headers };
    } catch (error) {
        console.error("Error loading CSV file:", error);
        return [];
    }
}

async function prepareRows(rowData, headers) {
    const rows = await Promise.all(rowData.map(async row => {
        const obj = {};
        headers.forEach((header, index) => {
            obj[header] = row[header].trim();
        });
        for (let key of BASELINEFILES.keys()) {
            // Create a new cell for each effective robustess value
            const wauc_ood = parseFloat(row[key].trim()) / 100;
            const wauc_id = parseFloat(row["FlyingThings3D"].trim()) / 100;

            const effrob = await calculateEffectiveRobustness(wauc_id, wauc_ood, BASELINEFILES.get(key)) * 100;
            const effrobRounded = Math.round(effrob * 100) / 100; // Round to 2 decimal places
            obj[key + "-robustness"] = effrobRounded;
        }
        return obj;
    }));
    return rows;
}


function makeFinalRow(headers, idprefix) {
    const tr = document.createElement("tr");
    tr.className += " has-background-yourmodel has-text-white";

    headers.forEach(header => {
        header = header.trim();
        const td = document.createElement("td");
        td.className += " has-background-yourmodel has-text-white";
        const key = header.replace("-robustness", "").trim();
        if (header === "Model") {
            const input = document.createElement("input");
            td.appendChild(input);
            input.type = "text";
            input.className = "input";
            input.placeholder = "Your Model";
            input.value = "Your Model";
            input.style.width = "110px";
        } else if (header === "FlyingThings3D") {
            const input = document.createElement("input");
            td.appendChild(input);
            input.type = "text";
            input.className = "input";
            input.placeholder = "N/A";
            input.style.width = "60px";
            input.id = idprefix + "-input-" + key;
            input.addEventListener("input", updateValues);
            if (key in YourModelWauc) {
                input.value = Math.round(YourModelWauc[key] * 100 * 1e3) / 1e3;
            } else {
                console.log(key, "not in", YourModelWauc)
            }
        } else if (!header.endsWith("-robustness")) {
            // td.className = "control";
            const input = document.createElement("input");
            td.appendChild(input);
            input.type = "text";
            input.className = "input";
            input.placeholder = "N/A";
            input.style.width = "60px";
            input.id = idprefix + "-input-" + key;
            input.addEventListener("input", updateValues);
            if (key in YourModelWauc) {
                input.value = Math.round(YourModelWauc[key] * 100 * 1e3) / 1e3;
            } else {
                console.log(key, "not in", YourModelWauc)
            }
        } else {
            td.id = idprefix + "-dynamic-" + key + "-robustness";

            if (key + "-robustness" in YourModelWauc) {
                td.textContent = Math.round(YourModelWauc[key + "-robustness"] * 100) / 100;
            } else {
                td.textContent = "TBD";
                console.log(key + "-robustness", "not in", YourModelWauc)
            }
        }
        tr.appendChild(td);
    });

    return tr;
}

async function updateValues(ev) {

    if (ev) {

        const el = ev.target;
        const id = ev.target.id

        try {
            if (id.startsWith('subset-')) {
                document.getElementById(id.replace("subset", "wide")).value = el.value;
            } else if (id.startsWith('wide-')) {
                document.getElementById(id.replace("wide", "subset")).value = el.value;
            } else {
                console.warn("Unknown ID", id, "for event", ev)
            }
        } catch { }
    }
    const element_id = document.getElementById("wide-input-FlyingThings3D")
    const wauc_id = parseFloat(element_id.value) / 100;
    YourModelWauc["FlyingThings3D"] = wauc_id;
    
    for (const key of BASELINEFILES.keys()) {
        console.log(key)
        try {
            const element_ood = document.getElementById("wide-input-" + key)
            if (!element_ood.value) {
                // console.log("-input-" + key, "does not exist")
                continue;
            }
            const wauc_ood = parseFloat(element_ood.value) / 100;

            const effrob = await calculateEffectiveRobustness(wauc_id, wauc_ood, BASELINEFILES.get(key)) * 100;
            const effrobRounded = Math.round(effrob * 100) / 100; // Round to 2 decimal places

            try {
                var x = document.getElementById("subset-dynamic-" + key + "-robustness");
                x.textContent = effrobRounded;
            } catch (e) {
                console.log(e);
            }

            try {
                var x = document.getElementById("wide-dynamic-" + key + "-robustness");
                x.textContent = effrobRounded;
            } catch (e) { 
                console.log(e);
            }

            YourModelWauc[key] = wauc_ood;
            YourModelWauc[key + "-robustness"] = effrob;

        } catch (error) {
            console.error("Error updating values:", error);
            var x = document.getElementById("dynamic-" + key + "-robustness");
            x.textContent = "Error";
        }
    }
    console.log("Update")

    await updateSideBySideContent()
}

async function displayCSVAsTable(elementId) {


    function makeTableHeader(headers) {
        const thead = document.createElement("thead");

        const firstHeaderRow = document.createElement("tr");
        const secondHeaderRow = document.createElement("tr");

        const l1 = headers.filter(header => header !== "Model" && header !== "FlyingThings3D" && !header.endsWith("-robustness")).length;
        const l2 = headers.filter(header => header.endsWith("-robustness")).length;
        if (l1 != l2) console.error("Header length mismatch");

        let prevFirstHeader = "tbd";

        headers.forEach((header, index) => {
            header = header.trim();
            const th0 = document.createElement("th");
            const th1 = document.createElement("th");

            let firstHeader;
            if (header === "Model") {
                firstHeader = "";
                th1.style.width = "150px";
                th1.textContent = "Model";
            } else if (header === "FlyingThings3D") {
                firstHeader = "WAUC [%] ID";
                th1.style.width = "150px";
                th1.textContent = "FlyingThings3D";
            } else if (!header.endsWith("-robustness")) {
                firstHeader = "WAUC [%] OOD";
                th1.style.width = "200px";
                th1.textContent = header;
            } else {
                firstHeader = "Effective Robustness wrt. WAUC [%]";
                th1.style.width = "200px";
                th1.textContent = header.replace("-robustness", "");
            }

            if (prevFirstHeader === firstHeader) {
                firstHeaderRow.lastChild.colSpan += 1;
            } else {
                const th0 = document.createElement("th");
                th0.textContent = firstHeader;
                th0.colSpan = 1;
                firstHeaderRow.appendChild(th0);
                prevFirstHeader = firstHeader;
            }
            secondHeaderRow.appendChild(th1);
        });

        thead.appendChild(firstHeaderRow);
        thead.appendChild(secondHeaderRow);

        return thead;
    }

    try {
        const selectDataset = document.getElementById('select-dataset-sidebyside');
        const selectCheckpoint = document.getElementById('select-checkpoint-sidebyside');

        const datasets = [
            'Sintel',
            'KITTI',
            'HD1K',
            'Driving',
            'VIPER',
            'Spring',
        ]
        const checkpoint = selectCheckpoint.value;

        let { data: data_merged, headers: headers_merged } = await loadCSV("./static/data/summary_metrics_merged.csv");

        data_merged = data_merged.filter(row => {
            return (// row["dataset"].trim() === dataset
                row["training_stage"].trim() === checkpoint
                && row["metric_name"].trim() === "WAUC");
        });

        console.log("Filtered data_merged", data_merged, "for", checkpoint);

        const headers = ["Model", "FlyingThings3D", ...datasets, ...datasets.map(s => s + "-robustness")];

        const nets = [... new Set(data_merged.map(row => row["net"]))];

        console.log("nets", nets);

        const rows = []

        for (const net of nets) {
            const data_net = data_merged.filter(row => row["net"] === net);

            const id = data_net.find(row => row["dataset"] === "FlyingThings3D");
            const wauc_id = parseFloat(id["metric_value"]);
            const wauc_id_percent = (wauc_id * 100).toFixed(2);
            const row = {
                "Model": net,
                "FlyingThings3D": wauc_id_percent
            }

            for (const dataset of datasets) {
                const ood = data_net.find(row => row["dataset"] === dataset);

                if (ood) {
                    const wauc_ood = parseFloat(ood["metric_value"]);
                    const wauc_ood_percent = (wauc_ood * 100).toFixed(2);
                    row[dataset] = wauc_ood_percent;

                    const wauc_id = parseFloat(id["metric_value"]);
                    const effrob = await calculateEffectiveRobustness(wauc_id, wauc_ood, BASELINEFILES.get(dataset)) * 100;
                    const effrobRounded = Math.round(effrob * 100) / 100; // Round to 2 decimal places
                    row[dataset + "-robustness"] = effrobRounded;
                } else {
                    row[dataset] = "N/A";
                    row[dataset + "-robustness"] = "N/A";
                }
            }
            rows.push(row);

        }
        console.log("Prepared rows", rows);

        // const rows = prepareRows(data_merged, headers_merged);


        // const {data: rowData, headers} = await loadCSV(filePath);
        console.log("headers", headers)

        // Parse the CSV content
        // const rows = csvText.split("\n").map(row => row.split(",")).filter(row => row.length > 1);

        // Create the HTML table
        const table = document.createElement("table");
        table.className = "table is-bordered is-striped is-hoverable is-fullwidth  has-sticky-column";

        // // Create the table header
        const thead = makeTableHeader(headers);
        table.appendChild(thead);

        // Create the table body
        const tbody = document.createElement("tbody");

        const tr = makeFinalRow(headers, 'wide');
        tbody.appendChild(tr);

        // const rows = await prepareRows(rowData, headers);
        console.log(rows);
        rows.forEach(row => {
            // Create a new row for each data entry
            const tr = document.createElement("tr");

            headers.forEach(key => {
                const td = document.createElement("td");
                td.textContent = row[key];
                tr.appendChild(td);
            });


            tbody.appendChild(tr);
        });


        console.log(rows);

        table.appendChild(tbody);


        // Append the table to the container
        const tableContainer = document.getElementById(elementId);
        tableContainer.innerHTML = "";
        tableContainer.appendChild(table);
    } catch (error) {
        console.error("Error loading CSV file:", error);
    }
}

async function displayCSVAsTableSubset(elementId) {


    function makeTableHeader(headers) {
        const thead = document.createElement("thead");

        const firstHeaderRow = document.createElement("tr");
        const secondHeaderRow = document.createElement("tr");

        const l1 = headers.filter(header => header !== "Model" && header !== "FlyingThings3D" && !header.endsWith("-robustness")).length;
        const l2 = headers.filter(header => header.endsWith("-robustness")).length;
        if (l1 != l2) console.error("Header length mismatch");

        headers.forEach((header, index) => {
            header = header.trim();
            const th0 = document.createElement("th");
            const th1 = document.createElement("th");
            if (header === "Model") {
                th0.textContent = "";
                th0.colSpan = 1;
                th1.style.width = "150px";
                th1.textContent = "Model";
            } else if (header === "FlyingThings3D") {
                th0.textContent = "WAUC [%] ID";
                th0.colSpan = 1;
                th1.style.width = "150px";
                th1.textContent = "FlyingThings3D";
            } else if (!header.endsWith("-robustness")) {
                th0.textContent = "WAUC [%] OOD";
                th0.colSpan = 1;
                th1.style.width = "200px";
                th1.textContent = header;
            } else {
                th0.textContent = "Effective Robustness wrt. WAUC [%]";
                th0.colSpan = 1;
                th1.style.width = "200px";
                th1.textContent = header.replace("-robustness", "");
            }
            firstHeaderRow.appendChild(th0);
            secondHeaderRow.appendChild(th1);
        });

        thead.appendChild(firstHeaderRow);
        thead.appendChild(secondHeaderRow);

        return thead;
    }

    try {
        const selectDataset = document.getElementById('select-dataset-sidebyside');
        const selectCheckpoint = document.getElementById('select-checkpoint-sidebyside');

        const dataset = selectDataset.value;
        const checkpoint = selectCheckpoint.value;

        let { data: data_merged, headers: headers_merged } = await loadCSV("./static/data/summary_metrics_merged.csv");

        data_merged = data_merged.filter(row => {
            return (// row["dataset"].trim() === dataset
                row["training_stage"].trim() === checkpoint
                && row["metric_name"].trim() === "WAUC");
        });

        console.log("Filtered data_merged", data_merged, "for", dataset, checkpoint);

        const headers = ["Model", "FlyingThings3D", dataset, dataset + "-robustness"];

        const nets = [... new Set(data_merged.map(row => row["net"]))];

        console.log("nets", nets);

        const rows = []

        for (const net of nets) {
            const data_net = data_merged.filter(row => row["net"] === net);

            const id = data_net.find(row => row["dataset"] === "FlyingThings3D");
            const ood = data_net.find(row => row["dataset"] === dataset);

            if (id && ood) {
                const wauc_id = parseFloat(id["metric_value"]);
                const wauc_ood = parseFloat(ood["metric_value"]);
                const effrob = await calculateEffectiveRobustness(wauc_id, wauc_ood, BASELINEFILES.get(dataset)) * 100;

                const wauc_id_percent = (wauc_id * 100).toFixed(2);
                const wauc_ood_percent = (wauc_ood * 100).toFixed(2);
                const effrobRounded = Math.round(effrob * 100) / 100; // Round to 2 decimal places

                rows.push({
                    "Model": net,
                    "FlyingThings3D": wauc_id_percent,
                    [dataset]: wauc_ood_percent,
                    [dataset + "-robustness"]: effrobRounded,
                });
            }
        }
        console.log("Prepared rows", rows);

        // const rows = prepareRows(data_merged, headers_merged);


        // const {data: rowData, headers} = await loadCSV(filePath);
        console.log("headers", headers)

        // Parse the CSV content
        // const rows = csvText.split("\n").map(row => row.split(",")).filter(row => row.length > 1);

        // Create the HTML table
        const table = document.createElement("table");
        table.className = "table is-bordered is-striped is-hoverable is-fullwidth";

        // // Create the table header
        const thead = makeTableHeader(headers);
        table.appendChild(thead);

        // Create the table body
        const tbody = document.createElement("tbody");

        const tr = makeFinalRow(headers, 'subset');
        tbody.appendChild(tr);

        // const rows = await prepareRows(rowData, headers);
        console.log(rows);
        rows.forEach(row => {
            // Create a new row for each data entry
            const tr = document.createElement("tr");

            headers.forEach(key => {
                const td = document.createElement("td");
                td.textContent = row[key];
                tr.appendChild(td);
            });


            tbody.appendChild(tr);
        });


        console.log(rows);

        table.appendChild(tbody);


        // Append the table to the container
        const tableContainer = document.getElementById(elementId);
        tableContainer.innerHTML = "";
        tableContainer.appendChild(table);
    } catch (error) {
        console.error("Error loading CSV file:", error);
    }
}

async function makePlot(xcol, ycol, elementId) {
    let { data: data_merged, headers: headers_merged } = await loadCSV("./static/data/summary_metrics_merged.csv");
    console.debug("summary_metrics_merged", data_merged);

    const data_things = data_merged.filter(row => row["training_stage"].trim() === "C+T");
    const data_sintel = data_merged.filter(row => row["training_stage"].trim() === "S");
    const data_kitti = data_merged.filter(row => row["training_stage"].trim() === "K");
    const data_other = data_merged.filter(row => (row["training_stage"] !== "C+T" && row["training_stage"] !== "S" && row["training_stage"] !== "K"));

    const plotData = PlotData;
    plotData.length = 0;

    for ({ data, training_stage, color, symbol } of [
        { data: data_things, training_stage: 'C+T', color: "blue", symbol: "circle" },
        { data: data_sintel, training_stage: 'S', color: "red", symbol: "square" },
        { data: data_kitti, training_stage: 'K', color: "orange", symbol: "diamond" },
        { data: data_other, training_stage: 'Other', color: "green", symbol: "cross" },
    ]) {
        const data_x = data.filter(row => {
            for (const key in xcol) {
                if (row[key] !== xcol[key]) {
                    return false;
                }
            }
            return true;
        });
        const data_y = data.filter(row => {
            for (const key in ycol) {
                if (row[key] !== ycol[key]) {
                    return false;
                }
            }
            return true;
        });

        if (data_x.length !== data_y.length) {
            console.error(`Mismatch in data length for ${xcol} and ${ycol}:`, data_x.length, data_y.length);
        }

        nets = data_x.map(row => { return { net: row["net"], checkpoint: row["checkpoint"] } });
        nets = nets.filter(net => data_y.some(row => row["net"] === net.net && row["checkpoint"] === net.checkpoint));

        let xArray = nets.map(({ net, checkpoint }) =>
            data_x.find(row => ((row['net'] === net) && (row['checkpoint'] === checkpoint)))
        );
        let yArray = nets.map(({ net, checkpoint }) =>
            data_y.find(row => ((row['net'] == net) && (row['checkpoint'] == checkpoint)))
        );

        xArray = xArray.map(row => parseFloat(row['metric_value']));
        yArray = yArray.map(row => parseFloat(row['metric_value']));

        // const yArray = nets.map(row => parseFloat(row[ycol]));

        plotData.push({
            x: xArray,
            y: yArray,
            text: nets.map(({ net, checkpoint }) => `${net}<br>${checkpoint.split('/').pop()}`), // shown on hover
            name: training_stage, // shown in legend
            mode: "markers",
            marker: { symbol: symbol, color: color },
            hovertemplate: `<b>%{text}</b><br>x: %{x:.2f}<br>y: %{y:.2f}<extra></extra>`,
        });

        console.debug(`Plotting ${training_stage} data:`, xArray, yArray);
    }

    console.log("plotData", plotData);
    // Define Layout
    const layout = {
        xaxis: { title: { text: `${xcol.metric_name} on ${xcol.dataset}` } },
        yaxis: { title: { text: `${ycol.metric_name} on ${ycol.dataset}` } },
        title: { text: `${xcol.metric_name} on ${xcol.dataset} vs. ${ycol.metric_name} on ${ycol.dataset}` },
    };

    // Display with Plotly
    Plot = await Plotly.newPlot(elementId, plotData, layout);

    console.log(`Plot created for ${xcol} vs ${ycol} in element ${elementId}`);
}

async function makePlotDynamic() {
    const xdataset = document.getElementById("x-select-dataset").value;
    const ydataset = document.getElementById("y-select-dataset").value;

    function getDataset(dataset) {
        const parts = dataset.split(" (");
        const parts2 = parts[1] ? parts[1].split(", ") : [];
        return {
            dataset: parts[0].trim(),
            dataset_stage: parts2[0] ? parts2[0].split(")")[0].trim() : "",
            dataset_pass: parts2[1] ? parts2[1].split(")")[0].trim() : "",
        };
    }
    const xcol = getDataset(xdataset);
    const ycol = getDataset(ydataset);


    xcol.metric_name = document.getElementById("x-select-metric").value;
    ycol.metric_name = document.getElementById("y-select-metric").value;

    console.log("xcol", xcol, "ycol", ycol);

    const elementId = "scatter-plot-custom";

    // // Clear the previous plot
    // const plotDiv = document.getElementById(elementId);
    // plotDiv.innerHTML = "";

    // Create the new plot
    Plot = await makePlot(xcol, ycol, elementId);
}

function makeYourModelPlotEntry(xdataset, ydataset) {

    return {
        x: [YourModelWauc[xdataset]],
        y: [YourModelWauc[ydataset]],
        text: "Your Model", // shown on hover
        name: "Your Model", // shown in legend
        mode: "markers",
        marker: { symbol: "star", color: "", size: 10 },
        //hovertemplate: `<b>%{text}</b><br>x: %{x:.2f}<br>y: %{y:.2f}<extra></extra>`,
        flag: "yourmodel"
    }
}

async function updateSideBySideContent() {
    console.log("updateSideBySideContent")
    displayCSVAsTableSubset("csv-table-container");
    const dataset = document.getElementById("select-dataset-sidebyside").value;
    await makePlot({ dataset: "FlyingThings3D", metric_name: "WAUC" }, { dataset: dataset, metric_name: "WAUC" }, "scatter-plot");

    const xdataset = "FlyingThings3D";
    const ydataset = dataset;
    console.log(YourModelWauc)
    if (xdataset in YourModelWauc && ydataset in YourModelWauc) {
        const idx = PlotData.findIndex(el => el.flag == "yourmodel");
        if (idx > -1) {
            PlotData.splice(idx, 1)
        }
        PlotData.push(makeYourModelPlotEntry(xdataset, ydataset))
        console.log("Redraw plot")
        Plotly.redraw(Plot)
    }
}

async function prepareGraphSelects() {
    const { data: data_merged, headers: headers_merged } = await loadCSV("./static/data/summary_metrics_merged.csv");

    metric_names = data_merged.map(row => row['metric_name'].trim());
    metric_names = [...new Set(metric_names)].sort(); // Remove duplicates

    datasets = data_merged.map(row => {
        if (row["dataset_pass"] == "") {
            return `${row['dataset'].trim()} (${row['dataset_stage'].trim()})`
        } else {
            return `${row['dataset'].trim()} (${row['dataset_stage'].trim()}, ${row['dataset_pass'].trim()})`
        }
    });
    datasets = [...new Set(datasets)].sort(); // Remove duplicates

    // select dataset
    for (const select of document.getElementsByClassName("select-dataset")) {
        select.innerHTML = ""; // Clear existing options
        for (const dataset of datasets) {
            const option = document.createElement("option");
            option.value = dataset;
            option.textContent = dataset;
            select.appendChild(option);

            if (select.id === "x-select-dataset" && dataset.startsWith("FlyingThings3D")) {
                option.selected = true; // Default selection for x-axis
            } else if (select.id === "y-select-dataset" && dataset.startsWith("Sintel")) {
                option.selected = true; // Default selection for y-axis
            }
        }
        select.addEventListener("change", makePlotDynamic);
        //select.innerHTML = datasets.map(dataset => `<option value="${dataset}">${dataset}</option>`).join("");
    }


    // select metric
    for (const select of document.getElementsByClassName("select-metric")) {
        select.innerHTML = ""; // Clear existing options
        for (const metric of metric_names) {
            const option = document.createElement("option");
            option.value = metric;
            option.textContent = metric;
            select.appendChild(option);

            option.selected = metric === "WAUC"; // Default selection for x-axis
        }
        select.addEventListener("change", makePlotDynamic);
    }
    console.debug("Done preapring");
}


async function prepareSideBySideSelects() {
    document.getElementById("select-dataset-sidebyside").addEventListener("change", () => updateSideBySideContent());
    document.getElementById("select-checkpoint-sidebyside").addEventListener("change", () => updateSideBySideContent());
}


async function prepare() {
    prepareGraphSelects();
    prepareSideBySideSelects();
}

async function openTab(tabId) {
    // Hide all tab contents
    const tabContents = document.querySelectorAll(".tab-content");
    tabContents.forEach(content => content.style.display = "none");

    // Remove 'is-active' class from all tabs
    const tabs = document.querySelectorAll(".tabs li");
    tabs.forEach(tab => tab.classList.remove("is-active"));

    // Show the selected tab content
    document.getElementById(tabId).style.display = "block";

    // Add 'is-active' class to the clicked tab
    const activeTab = document.querySelector(`.tabs li[data-tab="${tabId}"]`);
    if (activeTab) {
        activeTab.classList.add("is-active");
    }

    if (tabId === "tab-content-graph") {
        // If the plot tab is opened, update the plot
        makePlotDynamic();
    } else if (tabId === "tab-content-both") {
        makePlot({ dataset: "FlyingThings3D", metric_name: "WAUC" }, { dataset: "Sintel", metric_name: "WAUC" }, "scatter-plot");
        console.log("makePlot called");
    }

    document.getElementById("section-results").addEventListener("scroll", function () {
        var translate = "translate(0," + this.scrollTop + "px)";
        this.querySelector("thead").style.transform = translate;
    });
}

// Call the function with the path to your CSV file
document.addEventListener("DOMContentLoaded", () => {
    prepare() // prepare select boxes
    displayCSVAsTableSubset("csv-table-container"); // Update the path to your CSV file
    displayCSVAsTable("csv-table-container-wide"); // Update the path to your CSV file
    openTab("tab-content-both"); // Open the side-by-side tab by default

});
