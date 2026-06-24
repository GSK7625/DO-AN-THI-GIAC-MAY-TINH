import os
import pandas as pd
import xml.etree.ElementTree as ET
from xml.dom import minidom

def main():
    print("=== GENERATING REAL-WORLD ROUTE CONFIGURATION FOR SUMO ===")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(os.path.dirname(script_dir), 'input')
    
    tmc_path = os.path.join(input_dir, 'tmc.csv')
    tracks_path = os.path.join(input_dir, 'vehicle_tracks_xy.csv')
    output_rou_path = os.path.join(script_dir, 'configs', 'osm_cut_real.rou.xml')
    output_sumocfg_path = os.path.join(script_dir, 'configs', 'osm_cut_real.sumocfg')
    
    if not os.path.exists(tmc_path) or not os.path.exists(tracks_path):
        print("Error: Input files not found in input/ directory.")
        return
        
    print(f"Reading {tmc_path} and {tracks_path}...")
    tmc_df = pd.read_csv(tmc_path)
    tracks_df = pd.read_csv(tracks_path)
    
    # 1. Find the first frame for each track_id
    print("Calculating departure times from tracking frames...")
    first_frames = tracks_df.groupby('track_id')['frame'].min().reset_index()
    first_frames.rename(columns={'frame': 'first_frame'}, inplace=True)
    
    # 2. Merge with tmc_df to get routes, classes, and directions
    merged_df = pd.merge(tmc_df, first_frames, on='track_id', how='left')
    
    # Handle any potential missing first_frame (fill with 0 as fallback)
    merged_df['first_frame'] = merged_df['first_frame'].fillna(0).astype(int)
    
    # Calculate departure time in seconds (FPS = 30)
    merged_df['depart_time'] = merged_df['first_frame'] / 30.0
    
    # Sort vehicles by departure time to comply with SUMO's chronological ordering requirement
    merged_df.sort_values(by='depart_time', inplace=True)
    
    # 3. Define mapping of direction + movement to route_id
    # North input is 428067750#0
    # South input is -577951513
    # East input is 428067759#0
    # West input is 428067756.116
    route_mapping = {
        ('East', 'Straight'): 'E_W',
        ('East', 'Left Turn'): 'E_S',
        ('East', 'Right Turn'): 'E_N',
        ('East', 'Stationary'): 'E_W',
        
        ('West', 'Straight'): 'W_E',
        ('West', 'Left Turn'): 'W_N',
        ('West', 'Right Turn'): 'W_S',
        ('West', 'Stationary'): 'W_E',
        
        ('North', 'Straight'): 'N_S',
        ('North', 'Left Turn'): 'N_E',
        ('North', 'Right Turn'): 'N_W',
        ('North', 'Stationary'): 'N_S',
        
        ('South', 'Straight'): 'S_N',
        ('South', 'Left Turn'): 'S_W',
        ('South', 'Right Turn'): 'S_E',
        ('South', 'Stationary'): 'S_N',
    }
    
    # 4. Define class mapping
    class_mapping = {
        'car': 'standard_car',
        'truck': 'truck',
        'bus': 'truck',
        'motorcycle': 'motorcycle'
    }
    
    # Build XML document
    root = ET.Element('routes')
    root.set('xmlns:xsi', 'http://www.w3.org/2001/XMLSchema-instance')
    root.set('xsi:noNamespaceSchemaLocation', 'http://sumo.dlr.de/xsd/routes_file.xsd')
    
    # Append vehicle types (vTypes)
    vtypes = [
        ET.Element('vType', id='standard_car', accel='2.6', decel='4.5', length='4.5', minGap='2.0', maxSpeed='13.41',
                   sigma='0.5', speedFactor='1.0', jmIgnoreKeepClearTime='5', jmDriveAfterYellowTime='1.0', jmTimegapMinor='1.0'),
        ET.Element('vType', id='truck', accel='1.0', decel='4.0', length='10.0', minGap='2.5', maxSpeed='10.0',
                   sigma='0.3', speedFactor='0.9', jmIgnoreKeepClearTime='5', jmDriveAfterYellowTime='1.0', jmTimegapMinor='1.5'),
        ET.Element('vType', id='motorcycle', accel='2.5', decel='5.0', length='2.2', minGap='1.2', maxSpeed='13.41',
                   sigma='0.5', speedFactor='1.1', jmIgnoreKeepClearTime='3', jmDriveAfterYellowTime='1.0', jmTimegapMinor='0.8')
    ]
    for vt in vtypes:
        root.append(vt)
        
    # Append route definitions
    routes = [
        ET.Element('route', id='N_S', edges='428067750#0 428067754#0'),
        ET.Element('route', id='N_E', edges='428067750#0 378376408#0'),
        ET.Element('route', id='N_W', edges='428067750#0 160183267#0'),
        
        ET.Element('route', id='S_N', edges='-577951513 -428067750#1'),
        ET.Element('route', id='S_E', edges='-577951513 378376408#0'),
        ET.Element('route', id='S_W', edges='-577951513 160183267#0'),
        
        ET.Element('route', id='E_N', edges='428067759#0 -428067750#1'),
        ET.Element('route', id='E_S', edges='428067759#0 428067754#0'),
        ET.Element('route', id='E_W', edges='428067759#0 160183267#0'),
        
        ET.Element('route', id='W_N', edges='428067756.116 -428067750#1'),
        ET.Element('route', id='W_S', edges='428067756.116 428067754#0'),
        ET.Element('route', id='W_E', edges='428067756.116 378376408#0'),
    ]
    for r in routes:
        root.append(r)
        
    # Append vehicle instances
    print("Mapping vehicles to route types and timestamps...")
    mapped_count = 0
    ignored_count = 0
    
    for idx, row in merged_df.iterrows():
        track_id = row['track_id']
        direction = row['direction']
        movement = row['movement']
        vehicle_class = row['class']
        depart_time = row['depart_time']
        
        route_key = (direction, movement)
        route_id = route_mapping.get(route_key)
        
        if not route_id:
            print(f"Warning: No route mapping found for track {track_id} ({direction}, {movement}). Defaulting to Straight route.")
            # Fallback to straight
            route_id = route_mapping.get((direction, 'Straight'))
            
        vtype_id = class_mapping.get(vehicle_class, 'standard_car')
        
        veh_elem = ET.Element('vehicle')
        veh_elem.set('id', f'veh_{track_id}')
        veh_elem.set('type', vtype_id)
        veh_elem.set('route', route_id)
        veh_elem.set('depart', f"{depart_time:.2f}")
        veh_elem.set('departLane', 'best')
        veh_elem.set('departSpeed', 'desired')
        
        root.append(veh_elem)
        mapped_count += 1
        
    # Format and save XML
    xml_str = ET.tostring(root, encoding='utf-8')
    parsed_xml = minidom.parseString(xml_str)
    pretty_xml = parsed_xml.toprettyxml(indent="    ")
    
    # Remove empty lines that minidom sometimes inserts
    clean_xml = "\n".join([line for line in pretty_xml.splitlines() if line.strip()])
    
    with open(output_rou_path, 'w', encoding='utf-8') as f:
        f.write(clean_xml)
        
    print(f"Route file created successfully: {output_rou_path}")
    print(f"Total vehicles generated: {mapped_count}")
    
    # 5. Create the sumocfg configuration file
    print("Creating SUMO config file for real-world scenario...")
    sumocfg_content = """<?xml version="1.0" encoding="UTF-8"?>
<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
    xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">
    <input>
        <net-file value="osm_cut.net.xml"/>
        <route-files value="osm_cut_real.rou.xml"/>
        <additional-files value="osm_cut_rl.add.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="1000"/>
        <step-length value="0.1"/>
    </time>
    <processing>
        <!-- Teleport vehicles stuck longer than 120s to prevent permanent deadlock -->
        <time-to-teleport value="120"/>

        <!-- After 10s of waiting, ignore vehicles blocking the junction interior -->
        <ignore-junction-blocker value="10"/>
    </processing>
</configuration>
"""
    with open(output_sumocfg_path, 'w', encoding='utf-8') as f:
        f.write(sumocfg_content)
    print(f"SUMO config file created successfully: {output_sumocfg_path}")
    
    print("=== ROUTE GENERATION COMPLETED ===")

if __name__ == '__main__':
    main()
