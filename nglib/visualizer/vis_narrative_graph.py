######
# This file uses dash-cytoscape, a Python package for visualizing graph structures, to
# produce visualizations of graph salads.  A single file name (the graph salad to be visualized)
# is expected as an argument.
######

import dill
import dash
import random
from dash.dependencies import Input, Output
import dash_cytoscape as cyto
import dash_html_components as html
import sys

if len(sys.argv) == 1:
    file_name = 'C:/Users/davidcdwu/PycharmProjects/mcnc/ng_example3.p'
else:
    file_name = sys.argv[1]


ng = dill.load(open(file_name, 'rb'))
elements = []
for nid, node in ng.nodes.items():
    temp = {'data': dict()}
    temp['data']['id'] = nid
    temp['data']['ntag'] = node.ntag
    temp['data']['ntype'] = node.ntype  
    temp['data']['name'] = node.name
    temp['data']['connectedness'] = node.connectedness
    temp['data']['length_name'] = max((len(node.name) * 10), 50)
    elements.append(temp)

for edge in ng.edges:
    nid1, nid2, rtype = edge
    temp = {'data': dict()}
    temp['data']['rid'] = ng.rtypes[rtype]
    temp['data']['rtype'] = rtype
    temp['data']['source'] = nid1
    temp['data']['target'] = nid2
    elements.append(temp)


rel_colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(len(ng.rtypes))]

app = dash.Dash(__name__)

default_stylesheet = [
    {
        'selector': 'node[ntag="predicate"]',
        'style': {
            'shape': 'ellipse',
            'background-color': 'lightgreen',
            'border-style': 'solid',
            'border-width': '2',
            'label': 'data(name)',
            'width': 'data(length_name)',
            'height': '50',
            'text-halign': 'center',
            'text-valign': 'center'
        }
    },
    {
        'selector': 'node[ntag="token"]',
        'style': {
            'shape': 'ellipse',
            'background-color': 'yellow',
            'border-style': 'solid',
            'border-width': '2',
            'label': 'data(name)',
            'width': 'data(length_name)',
            'height': '50',
            'text-halign': 'center',
            'text-valign': 'center'
        }
    },
    {
        'selector': 'node[connectedness=1]',
        'style': {
            'shape': 'ellipse',
            'background-color': 'cyan',
            'border-style': 'solid',
            'border-width': '2',
            'label': 'data(name)',
            'width': 'data(length_name)',
            'height': '50',
            'text-halign': 'center',
            'text-valign': 'center'
        }
    },
    {
        'selector': 'node[connectedness=2]',
        'style': {
            'shape': 'ellipse',
            'background-color': 'orange',
            'border-style': 'solid',
            'border-width': '2',
            'label': 'data(name)',
            'width': 'data(length_name)',
            'height': '50',
            'text-halign': 'center',
            'text-valign': 'center'
        }
    },
    {
        'selector': 'edge[rid=0]',
        'style': {
            'line-color': 'red',
            'curve-style': 'unbundled-bezier',
            'mid-target-arrow-shape': 'triangle',
            'mid-target-arrow-color': 'red',
            'arrow-scale': '1',
            'width': '1'
        }
    },
    {
        'selector': 'edge[rid=1]',
        'style': {
            'line-color': 'blue',
            'curve-style': 'unbundled-bezier',
            'mid-target-arrow-shape': 'triangle',
            'mid-target-arrow-color': 'blue',
            'arrow-scale': '1',
            'width': '1'
        }
    }
]

for ridx in range(2, len(ng.rtypes)):
    default_stylesheet.append({
        "selector": 'edge[rid={}]'.format(ridx),
        'style': {
            'line-color': rel_colors[ridx],
            'curve-style': 'unbundled-bezier',
            'mid-target-arrow-shape': 'triangle',
            'mid-target-arrow-color': rel_colors[ridx],
            'arrow-scale': '1',
            'width': '1'
        }
    })

app.layout = html.Div([
    cyto.Cytoscape(
        id='cytoscape',
        layout={'name': 'cose', 'nodeOverlap': '2500', 'randomize': 'false', 'nodeRepulsion': '1500'},
        elements=elements,
        style = {'width': '100%', 'height': '100%', 'position': 'absolute', 'top': '0px', 'left': '0px'},
    )
])

@app.callback(Output('cytoscape', 'stylesheet'),
              [Input('cytoscape', 'mouseoverEdgeData'),
               Input('cytoscape', 'mouseoverNodeData'),
               Input('cytoscape', 'tapNodeData')])
def generate_stylesheet(data_edge, data_node_hover, data_node_tap):
    if not (data_edge and data_node_hover and data_node_tap):
        return default_stylesheet

    stylesheet = default_stylesheet
    return stylesheet

if __name__ == '__main__':
    app.run_server(debug=True, port=8090)
