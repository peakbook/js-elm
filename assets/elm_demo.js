"use strict"
var svg;
var board;
var points=[];
var grids=[];
var grp_base;
var grp_pts;
var grp_grids;
var nclass = 4;
var width = 400;
var height = width;
var box = 10;
var elm = null;
var rnd = 0;

var colors = {"type_a":'#cc3333', "type_b":'#33cc33', "type_c":'#3333cc', "type_d":'#b8860b'};
var typint = {"type_a":0, "type_b":1, "type_c":2, "type_d":3};
var intcol = {"-1": '#ffffff', 0:'#cc3333', 1:'#33cc33', 2:'#3333cc', 3:'#b8860b'};

var kernels = {
    "sigmoid": MappingFuncs.Sigmoid,
    "hardlimit": MappingFuncs.HardLimit,
    "hyperbolictangent": MappingFuncs.HyperbolicTangent,
    "multiquadric": MappingFuncs.Multiquadric,
    "gaussian": MappingFuncs.Gaussian,
    "cosine": MappingFuncs.Cosine
};

function init()
{
    svg = d3.select("#canvas").append("svg");

    svg.attr("width", width)
       .attr("height",height);    
    svg.on("click", create_point);

    grp_base = svg.append("g")
        .attr("class","base")
        .append("rect")
        .attr("width", width)
        .attr("height",height)
        .attr("fill", "#ffffff");

    grp_grids = svg.append("g")
        .attr("class","grids");

    grp_pts = svg.append("g")
        .attr("class","points");

    var W = width/box;
    var H = height/box;
    grids = numeric.random([W*H, 3]);
    var idx=0;
    for (var j=0; j < H; j++) {
        for (var i=0; i < W; i++) {
            grids[idx][0] = i/W;
            grids[idx][1] = j/H;
            grids[idx][2] = -1;
            idx++;
        }
    }

    grp_grids.selectAll("rect").data(grids)
        .enter().append("rect")
        .attr("x", function(d){return d[0]*width;})
        .attr("y", function(d){return d[1]*height;})
        .attr("stroke", 'transparent')
        .attr("opacity", 0.5)
        .attr("width", box)
        .attr("height", box);

    onchange_alpha();
}

function onclick_train()
{
    var L = parseInt(document.getElementById('L').value, 10);
    var alpha = parseFloat(document.getElementById('alpha').value);
    var kernel = kernels[document.getElementById('kernel').value];
    var d = 2; 
    var N = points.length;
    var X = numeric.mul(1.0/width, numeric.getBlock(points, [0,0],[N-1,1]));
    var T = numeric.getBlock(points, [0,2],[N-1,2]);

    alpha = calc_alpha(alpha);

    elm = new ELM(d,L,nclass,alpha,kernel,ELMType.Classification);
    elm.train(X,T);

    for(var i=0;i<numeric.dim(grids)[0];i++)
    {
        grids[i][2] = elm.predict([[grids[i][0],grids[i][1]]])[0][0];
    }

    update_view();
}

function get_sample_type()
{
    var form = document.getElementById('sampleclass');
    var name = "types";
    var val;
    var radios = form.elements[name];

    for (var i=0, len=radios.length; i<len; i++) {
        if ( radios[i].checked ) { 
            val = radios[i].value;
            break; 
        }
    }
    return val; 
}

function create_point()
{
    var x = d3.mouse(this)[0];
    var y = d3.mouse(this)[1];
    var label = get_sample_type();

    points.push([x,y,typint[label]]);

    update_view();
}

function update_view()
{
    var obj = grp_pts.selectAll("circle").data(points);
    obj.enter().append("circle")
        .attr("cx", function(d){return d[0];})
        .attr("cy", function(d){return d[1];})
        .attr("r", 3)
        .attr("fill", function(d){return intcol[d[2]];});

    obj.exit().remove();

    grp_grids.selectAll("rect").data(grids)
        .attr("fill", function(d){
            return intcol[d[2]];
        });
}

function onclick_clear_samples()
{
    points = [];
    for(var i of grids) i[2] = -1;

    update_view();
}

function onclick_random_samples()
{
    onclick_clear_samples();

    switch (rnd) {
        case 0:
            for(var i=1;i<10;i++){
                [[0.05,0.05,0],[0.05,0.55,1],[0.55,0.05,2],[0.55,0.55,3]].forEach(function(x){
                    points.push([(Math.random()*0.4+x[0])*width,(Math.random()*0.4+x[1])*height, x[2]]);
                });
            }
            rnd++;
            break;
        case 1:
            for(var i=1;i<10;i++){
                [[0.30,0.40,0],[0.20,0.80,1]].forEach(function(d){
                    var theta = Math.random()*Math.PI*2;
                    var m = (Math.random()-0.5)*d[0];
                    var x = Math.cos(theta)*0.5+m;
                    var y = Math.sin(theta)*0.5+m;
                    points.push([(0.5+d[1]*x)*width, (0.5+d[1]*y)*height, d[2]]);
                });
            }
            rnd++;
            break;
        default:
            for(var i=1;i<10;i++){
                [[0.1,0.1,0],[0.35,0.1,1],[0.6,0.1,2],[0.85,0.1,3]].forEach(function(d){
                    points.push([(d[0]+d[1]*Math.random())*width, Math.random()*height, d[2]]);
                });
            }
            rnd=0;
    }
    update_view();
}

function onchange_alpha()
{
    var alpha = parseFloat(document.getElementById('alpha').value);
    alpha = calc_alpha(alpha);
    document.getElementById('alpha_val').textContent = alpha;
}

function calc_alpha(alpha)
{
    return alpha == 0.0 ? alpha: Math.pow(10,-alpha);
}
