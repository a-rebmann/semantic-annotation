import json
import os
from preprocessing.preprocessor import preprocess_label
from readwrite.loader import deserialize_data
from readwrite.writer import serialize_data
from const import BPMAI, TERMS_FOR_MISSING


def extract_resources_from_jsons(config):
    json_dir = config.resource_dir + "bpmai/models/"
    resource_labels = deserialize_data(config.resource_dir, BPMAI)
    if resource_labels is False:
        resource_labels = set()
        json_files = [f for f in os.listdir(json_dir) if f.endswith(".json") and not f.endswith("meta.json") and is_english_bpmn(json_dir, f)]
        json_files.sort()
        print("Total number of json files:", len(json_files))
        for jf in json_files:
            resource_labels.update([res for res in load_JSON(os.path.join(json_dir, jf))[3] if "glossary" not in res and res not in TERMS_FOR_MISSING])
        serialize_data(config.resource_dir, resource_labels, BPMAI)
    print(resource_labels)
    return resource_labels

def extract_service_tasks_from_jsons(config):
    json_dir = config.resource_dir + "bpmai/models/"
    task_labels = False
    #task_labels = deserialize_data(config.resource_dir, BPMAI)
    if task_labels is False:
        task_labels = set()
        json_files = [f for f in os.listdir(json_dir) if f.endswith(".json") and not f.endswith("meta.json") and is_english_bpmn(json_dir, f)]
        json_files.sort()
        print("Total number of json files:", len(json_files))
        for jf in json_files:
            task_labels.update([res for res in load_JSON(os.path.join(json_dir, jf))[2] if "glossary" not in res and res not in TERMS_FOR_MISSING])
        #serialize_data(config.resource_dir, task_labels, BPMAI)
    print(task_labels)
    return task_labels


def load_JSON(path_to_json):
    with open(path_to_json, 'r') as f:
        data = f.read()
        json_data = json.loads(data)
        if "childShapes" not in json_data.keys():
            print("no elements in "+path_to_json)
            return {}, {}, set(), set()
        follows, labels, tasks, pools_and_lanes = process_shapes(json_data['childShapes'])
        # sanitize all task labels
        for task in tasks:
            labels[task] = preprocess_label(labels[task])
        return follows, labels, tasks, pools_and_lanes


def is_english_bpmn(path_to_directory, json_file):
    # Checks whether considered json file is an English BPMN 2.0 .model according to meta file
    json_file = json_file.replace(".json", ".meta.json")
    with open(os.path.abspath(path_to_directory) + "/" + json_file, 'r') as f:
        data = f.read()
        json_data = json.loads(data)
    mod_language = json_data['model']['modelingLanguage']
    nat_language = json_data['model']['naturalLanguage']
    if mod_language == "bpmn20" and nat_language == "en":
        return True
    else:
        return False


def process_shapes(shapes):
    follows = {}
    labels = {}
    tasks = set()
    pools_and_lanes = set()

    # Analyze shape list and store all shapes and activities
    # PLEASE NOTE: the code below ignores BPMN sub processes
    for shape in shapes:

        # Save all shapes to dict
        # print(shape['stencil']['id'], shape)

        # If current shape is a pool or a lane, we have to go a level deeper
        if shape['stencil']['id'] == 'Pool' or shape['stencil']['id'] == 'Lane':
            if "name" in shape['properties'].keys():
                pools_and_lanes.add(shape['properties']['name'].replace('\n', ' ').replace('\r', '').replace('  ', ' '))
            result = process_shapes(shape['childShapes'])
            follows.update(result[0])
            labels.update(result[1])
            tasks.update(result[2])

        shapeID = shape['resourceId']
        outgoingShapes = [s['resourceId'] for s in shape['outgoing']]
        if shapeID not in follows:
            follows[shapeID] = outgoingShapes

        # Save all tasks and respective labels separately
        if shape['stencil']['id'] == 'Task':
            if 'tasktype' in shape and shape['properties']['tasktype'] != 'None':

                print(shape['properties']['tasktype'])
            if not shape['properties']['name'] == "":
                tasks.add(shape['resourceId'])
                labels[shape['resourceId']] = shape['properties']['name'].replace('\n', ' ').replace('\r', '').replace('  ', ' ')
            else:
                labels[shape['resourceId']] = 'Task'
        else:
            if 'name' in shape['properties'] and not shape['properties']['name'] == "":
                labels[shape['resourceId']] = shape['stencil']['id'] + " (" + shape['properties']['name'].replace('\n', ' ').replace('\r', '').replace('  ', ' ') + ")";
            else:
                labels[shape['resourceId']] = shape['stencil']['id']
    return follows, labels, tasks, pools_and_lanes