import carla
with open("gt_check.yaml", "r") as f:
    a = ["vehicle.mercedes.sprinter", "vehicle.mitsubishi.fusorosa", "vehicle.carlamotors.european_hgv"]
    for i in f:
        a += [i]
    a = " ".join(a)
# Connect to the client and retrieve the world object
client = carla.Client('carla', 2000)
client.set_timeout(120)
world = client.get_world()
#client.load_world('Town03')
blueprint_library = world.get_blueprint_library()

# Отфильтруйте и выведите ВСЕ blueprints для транспортных средств
bps = blueprint_library.filter('*')
k = 0
for i in bps:
    if 'vehicle' in i.tags and i.id not in a:
        k += 1
        print(f"""
    - name: car{40 + k}
      spawn_position: [{80 + 5 * k}, 12, 1, 0, 0, 0]
      id: {138 + k}
      model: "{i.id}"
      color: [{(110 + 10 * k) % 250}, 10, 231]
              """)
k = 0
for i in bps:
    if 'walker' in i.tags or 'pedestrian' in i.tags:
        k += 1
        print(f"""
    - name: pedestrian{ k}
      spawn_position: [{115 + 2 * k}, 12, 1, 0, 0, 0]
      id: {141 +  k}
      model: "{i.id}"
      color: [{(170 + 10 * k) % 250}, 10, 231]""")
"""from opencda.scenario_testing.utils.sim_api import multi_class_vehicle_blueprint_filter as filter
import json

with open("opencda/assets/blueprint_meta/bbx_stats_0915.json") as f:
    '''bp_meta = json.load(f)
    classes = {
                0: "car",
                1: "truck",
                2: "suv",
                3: "bus",
                4: "motorcycle",
                5: "bicycle",
            }

    for i in range(6):
        
        print(classes[i], filter(i, world.get_blueprint_library(), bp_meta))'''
    
    k = 0
    for i in f:
        if "vehicle" in i:
            print(f'''
    - name: cav{2 + k}
      spawn_position: [{29 + 5 * k}, 3, 1, 0, 0, 0]
      destination: [{29 + 5 * k}, 3, 0]
      id: {101 + k}
      model: "{i.split('"')[1]}"
      color: [{10 * k}, 10, 231]''')
            k += 1"""