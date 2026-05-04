# import networkx as nx
from pyvis.network import Network
import gravis as gv
import torch

def visualize(data,s_ntype, 
                  comp_g_src_nid, 
                  t_ntype, 
                  comp_g_tgt_nid,feat_nodes,html_file,include_edge_weight=True, normalize_weights=True,include_self_loops = False):
    '''ensure that html file in argument given doesnt already exist
    If include_edge_weight is True make sure to give edge weights as an edge attribute of comp_g called 'mask'
    i.e, comp_g[edge_type]["mask"][i] should have weight of ith edge of type edge_type
    '''
    
    colors ={}
    colors["cases"] = '#9999FF'
    colors["facts"] = '#00CC66'
    colors["terms"] = '#CCCC00'
    colors["articles"] = '#FF0000'
    colors["notes"] = '#9999FF'
    colors["icds"] = '#FF0000'
    x_coords ={}
    x_coords["cases"] = -360
    x_coords["facts"] = -120
    x_coords["terms"] = 120
    x_coords["articles"] = 360
    x_coords["notes"] = -360
    x_coords["icds"] = 360
    graph = {
    'graph':{
        'directed': True,
        'metadata': {
            'arrow_size':6,
            
            'edge_size': 0.3,
            'edge_label_size': 5,
            'edge_opacity':0.7,
            'node_size': 18,
            
        },
        'nodes': {
            
            
        },
        'edges': [
            
        ]
    }
    }
    for node_type in data.node_types:
        for i in range(data[node_type].x.size(0)):
            graph['graph']['nodes'][f'{node_type}_{i}']={}
            graph['graph']['nodes'][f'{node_type}_{i}']['metadata'] = {}
            graph['graph']['nodes'][f'{node_type}_{i}']['metadata']['shape'] = 'circle'
            graph['graph']['nodes'][f'{node_type}_{i}']['metadata']['color'] = colors[node_type]
            # graph['graph']['nodes'][f'{node_type}_{i}']['metadata']['x'] = x_coords[node_type]
            graph['graph']['nodes'][f'{node_type}_{i}']['label'] = feat_nodes[(node_type,i)]
            if (node_type == s_ntype and i == comp_g_src_nid) or (node_type == t_ntype and i == comp_g_tgt_nid):
                graph['graph']['nodes'][f'{node_type}_{i}']['metadata']['size'] = 45
                graph['graph']['nodes'][f'{node_type}_{i}']['metadata']['y'] = 0
                if(node_type == s_ntype):
                    graph['graph']['nodes'][f'{node_type}_{i}']['metadata']['x'] = -200
                    

                else:
                    graph['graph']['nodes'][f'{node_type}_{i}']['metadata']['x'] = 200

    for edge_type in data.edge_types:
        seen = [] # nodes for which neighbor edge softmax has already been calculated
        
        src_type, rel_type, dst_type = edge_type
        if include_self_loops or "self" not in rel_type:
            edge_index = data[edge_type].edge_index
            for i in range(edge_index.size(1)):
                src, dst = edge_index[:, i].tolist()
                temp = {}
                temp['source'] = f'{src_type}_{src}'
                temp['target'] = f'{dst_type}_{dst}'
                if normalize_weights:
                    # conduct softmax on its neighbors
                    if src not in seen:
                        req_indices = (edge_index[0] == src).nonzero()
                        values = torch.tensor([data[edge_type].mask[j] for j in req_indices])
                        softmax_values = torch.softmax(values, dim=0)
                        k = 0
                        for j in req_indices:
                            data[edge_type].mask[j] = round(softmax_values[k].item(), 2)
                            k += 1
                        del k, softmax_values, req_indices, values
                        seen.append(src)  
                if include_edge_weight:  
                    temp['label'] = data[edge_type]['mask'][i]
                    # temp['metadata']={}
                    # temp['metadata']['opacity'] = data[edge_type]['mask'][i]
                
                graph['graph']['edges'].append(temp)
                
                
                # if src.item() not in seen:
                #     req_indices = (edge_index[0] == src.item()).nonzero()
                #     values = data[edge_type].edge_weight[req_indices]
                #     softmax_values = torch.softmax(values, dim=0)
                #     data[edge_type].edge_weight[req_indices] = softmax_values
                #     seen.append(src.item())
     
    graph['graph']['metadata']['node_click'] = '$label'
    if include_edge_weight:
        graph['graph']['metadata']['edge_click'] = 'Edge weight $label'
    # print(json.dumps(graph, indent=4))
    fig = gv.d3(graph,show_node_label=False,node_hover_neighborhood=True,show_details=True,graph_height=500,details_height=200,show_menu_toggle_button=True,edge_label_data_source='label',show_edge_label=include_edge_weight,edge_curvature=0.06)#,layout_algorithm_active=True,use_many_body_force=False,use_links_force=False,use_collision_force=True,collision_force_radius=10,collision_force_strength=1,use_centering_force=False)
    
    fig.export_html(html_file)
    # fig.display()

    legend_html = """
    <div style='position: absolute; top: 10px; right: 10px; background-color: white; padding: 10px; border: 1px solid black;'>
        <strong>Legend</strong>
        <ul>
    """ + "".join([f"<li style='color:{color}'>{node_type}</li>" for node_type, color in colors.items()]) + "</ul></div>"
    # Insert the node features table into the HTML file
    with open(html_file, 'r') as file:
        html_content = file.read()

   
    insert_pos = html_content.find('</html>')

    # Insert the features table before the closing body tag
    html_content = html_content[:insert_pos] + legend_html + html_content[insert_pos:]

    # Write the modified content back to the HTML file
    with open(html_file, 'w') as file:
        file.write(html_content)

    print(f"Graph saved to {html_file}")

# def to_networkx_hetero(data,feat_nodes):
#     # G = nx.Graph()
#     new_feat_nodes={}
#     G = nx.DiGraph()
#     # Add nodes
#     for node_type in data.node_types:
#         for i in range(data[node_type].x.size(0)):
#             G.add_node(f'{node_type}_{i}', type=node_type)
#             new_feat_nodes[f'{node_type}_{i}'] = feat_nodes[(node_type,i)]

#     # Add edges
#     for edge_type in data.edge_types:
#         src_type, rel_type, dst_type = edge_type
#         edge_index = data[edge_type].edge_index
#         for i in range(edge_index.size(1)):
#             src, dst = edge_index[:, i].tolist()
#             G.add_edge(f'{src_type}_{src}', f'{dst_type}_{dst}', type=rel_type)

#     return G,new_feat_nodes

# Extract node features as an HTML table


# def visualize(graph):
    

#     G = to_networkx_hetero(graph)



    
#     net = Network()

#     net.from_nx(G)
#     for node in net.nodes:
#         node['shape'] = 'box'  
#         # node['size'] = 10
#         # node['borderWidth'] = 0  # Remove border
#     # Generate the initial HTML file
#     html_file = "heterogeneous_graph.html"
#     net.show(html_file)



    # features_table = generate_features_table(graph)

    # # Insert the node features table into the HTML file
    # with open(html_file, 'r') as file:
    #     html_content = file.read()

    # # Find the location to insert the table, e.g., after the main div
    # insert_pos = html_content.find('</body>')

    # # Insert the features table before the closing body tag
    # html_content = html_content[:insert_pos] + features_table + html_content[insert_pos:]

    # # Write the modified content back to the HTML file
    # with open(html_file, 'w') as file:
    #     file.write(html_content)

    # Inform the user
    # print(f"Graph with node features saved to {html_file}")
# def visualize_hetero_graph(data,feat_nodes):
#     net = Network(directed=True)#,layout=True)
#     net.force_atlas_2based(spring_length=200)
    

#     node_id = 0
#     node_mapping = {}
#     # Iterate over node types and add nodes
#     colors ={}
#     colors["cases"] = '#9999FF'
#     colors["facts"] = '#00CC66'
#     colors["terms"] = '#CCCC00'
#     colors["articles"] = '#FF0000'
#     levels ={}
#     levels["cases"] = 1
#     levels["facts"] = 2
#     levels["terms"] = 3
#     levels["articles"] = 4

#     for node_type in data.node_types:
#         for i, node in enumerate(data[node_type].x):
#             node_mapping[(node_type, i)] = node_id
#             net.add_node(node_id, 
#                          label=f'{node_type}_{i}', 
#                          shape='elipse',
#                          color=colors[node_type],
#                          title=feat_nodes[(node_type,i)],
#                          level = levels[node_type]
                         
                         
#                          )
#             node_id += 1
    
    
#     # Iterate over edge types and add edges
#     for edge_type in data.edge_types:
        
#         src_type, rel_type, dst_type = edge_type
#         for src, dst in data[edge_type].edge_index.t().tolist():
            
#             net.add_edge(node_mapping[(src_type, src)], node_mapping[(dst_type, dst)])
    
    
#     # Generate and show the visualization
#     html_file = "heterogeneous_graph.html"
#     net.toggle_stabilization(True)
#     # net.toggle_physics(False)
#     net.show(html_file,notebook=False)
#     def generate_features_table():
#         html_table = "<h3>Node Features</h3><table border='1'>"
#         for node_type in data.node_types:
#             html_table += f"<tr><th colspan='2'>{node_type}</th></tr>"
#             html_table += "<tr><th>Node</th><th>Features</th></tr>"
#             features = data[node_type].x
#             for i, feature in enumerate(features):
#                 html_table += f"<tr><td>{node_type}_{i}</td><td>{feat_nodes[(node_type,i)]}</td></tr>"
#         html_table += "</table>"
#         return html_table
#     features_table = generate_features_table()

#     # Insert the node features table into the HTML file
#     with open(html_file, 'r') as file:
#         html_content = file.read()

#     # Find the location to insert the table, e.g., after the main div
#     insert_pos = html_content.find('</body>')

#     # Insert the features table before the closing body tag
#     html_content = html_content[:insert_pos] + features_table + html_content[insert_pos:]

#     # Write the modified content back to the HTML file
#     with open(html_file, 'w') as file:
#         file.write(html_content)
#     # Inform the user
#     print(f"Graph saved to {html_file}")
