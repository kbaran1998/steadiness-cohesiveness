import React from 'react';
import { useRef, useEffect } from 'react'
import * as d3 from 'd3';
import { path } from 'd3';

function FimifMap(props) {


    let jsonFileName = props.dataset + "_" + props.method + ".json";
    let data = require("../json/" + jsonFileName);
    const embeddedData = data.map((d, i) => {
        if(d.emb.length === 2) d.emb.push(i);   // avoid double pushing (due to rendering issue)
        return d.emb;
    });

    let pathFileName = props.dataset + "_" + props.method + "_path.json";
    let pathData = require("../json/" + pathFileName);
    console.log(pathData)

    const xS = useRef();   // xScale
    const yS = useRef();   // yScale
    const cS = useRef();   // colorScale

    useEffect(() => { 

        const width = props.width;
        const height = props.height;
        const margin = { hor: props.width / 20, ver: props.height / 20 };

        const [minX, maxX] = d3.extent(embeddedData, d => d[0]);
        const [minY, maxY] = d3.extent(embeddedData, d => d[1]);

        const xScale = d3.scaleLinear()
                         .domain([minX, maxX])
                         .range([0, width]);
        
        const yScale = d3.scaleLinear()
                         .domain([minY, maxY])
                         .range([0, height]);

        const colorDomain = [...Array(props.labelNum).keys()]
        // console.log(colorDomain);
        const colorRange = ["#4e79a7","#f28e2c","#76b7b2","#59a14f","#edc949","#af7aa1","#ff9da7","#9c755f","#bab0ab", "#e15759","#5E4FA2"];
        // const colorRange = ["#3288BD", "#5E4FA2", "#66C2A5", "#ABDDA4", "#E6F598", "#FFFFBF", "#FEE08B", "#FDAE61", "#F46D43", "#D53E4F", "#9E0142"];
        const colorScale =  d3.scaleOrdinal()
            .domain(colorDomain)
            .range(colorRange);
        cS.current = colorScale;

        xS.current = xScale;
        yS.current = yScale;
        
        
        const radius = 3;

        // ANCHOR SVG width / height setting
        const svg = d3.select("#scatterplot" + props.dataset + props.method)
                      .attr("width", width + margin.hor * 2)
                      .attr("height", height + margin.ver * 2)
                        .append("g")
                        .attr("transform", "translate(" + margin.hor + ", " + margin.ver + ")");
        
        // ANCHOR Scatterplot rending & interaction
        svg.selectAll("circle")
           .data(embeddedData)
           .join(
               enter => {
                   enter.append("circle")
                        .attr("fill", d => {
                            if(props.isLabel) {
                                return colorScale(data[d[2]].label);
                            }   
                            else return "blue";
                        })
                        .attr("cx", d => xScale(d[0]))
                        .attr("cy", d => yScale(d[1]))
                        .style("opacity", 0.3)
                        .attr("r", radius);
               }
           );
        
        // svg.append("path")
        //    .data([points])
        //    .attr("d", d3.svg.line()
        //    .tension(0) // Catmull–Rom
        //    .interpolate("cardinal-closed"));
        // let line = d3.line()
        //             .x(d => d.x) // set the x values for the line generator
        //             .y(d => d.y) // set the y values for the line generator 
        //             .curve(d3.curveMonotoneX) // apply smoothing to the line

        let line = d3.line(d => xScale(d.x), d => yScale(d.y)).curve(d3.curveMonotoneX)
                     
        
        // console.log(pathData[0])
        // // for (let i = 0; i < pathData.length; i++) {
        //     // console.log(pathData[i])
        //     console.log(pathData)

        for (let i = 0; i < pathData.length; i++) {
            let test = pathData[i].map(d => {
                return {
                    x: d[0],
                    y: d[1]
                }
            });
            console.log(test)
            svg.append("path")
               .attr("d", line(test.slice(0, 4)))
               .attr("stroke", "blue")
               .attr("stroke-width", 3)
               .attr("fill", null)
               .style("opacity", 0.3);
               
        }
    });

    return (
        <div style={{width: props.width * 1.1, margin: 40}}>
            <svg id={"scatterplot" + props.dataset + props.method}></svg>
        </div>
    )

}

export default FimifMap;