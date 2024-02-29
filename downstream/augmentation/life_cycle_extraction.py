from const import ConceptType, type_mapping, XES_NAME
from model.augmented_log import AugmentedLog
from preprocessing.preprocessor import preprocess_label


def get_sub_logs_for_bo(res_df):
    obj = list(res_df[type_mapping[ConceptType.BO_NAME.value]+':0'].unique())
    logs = {}
    for o in obj:
        o = str(o)
        try:
            logs[o] = res_df[res_df[type_mapping[ConceptType.BO_NAME.value]+':0'].str.contains(o, na=False)]
        except TypeError:
            print("Cannot handle a business object")
    return logs


def find_common_components(tagged_set: object, ld: AugmentedLog, component_type: str = ConceptType.BO_NAME.value) -> tuple:
    components = {component_type: {}, ConceptType.ACTION_NAME.value: {}, ConceptType.ACTOR_NAME.value: {}, ConceptType.PASSIVE_NAME.value: {}}
    for tagged_label_key, tagged_label_value in tagged_set.items():
        components[component_type][tagged_label_key] = []
        components[ConceptType.ACTION_NAME.value][tagged_label_key] = []
        components[ConceptType.ACTOR_NAME.value][tagged_label_key] = []
        components[ConceptType.PASSIVE_NAME.value][tagged_label_key] = []
        curr_list = []
        for i in range(0, len(tagged_label_value[1])):
            if tagged_label_value[1][i] == component_type:
                curr_list.append((tagged_label_value[0][i]))
            else:
                if len(curr_list) > 0:
                    components[component_type][tagged_label_key].append(curr_list.copy())
                    curr_list = []
        if len(curr_list) > 0:
            components[component_type][tagged_label_key].append(curr_list.copy())
        curr_list = []
        for i in range(0, len(tagged_label_value[1])):
            if tagged_label_value[1][i] == ConceptType.ACTION_NAME.value:
                curr_list.append((tagged_label_value[0][i]))
            else:
                if len(curr_list) > 0:
                    components[ConceptType.ACTION_NAME.value][tagged_label_key].append(curr_list.copy())
                    curr_list = []
        if len(curr_list) > 0:
            components[ConceptType.ACTION_NAME.value][tagged_label_key].append(curr_list.copy())
        curr_list = []
        for i in range(0, len(tagged_label_value[1])):
            if tagged_label_value[1][i] == ConceptType.PASSIVE_NAME.value:
                curr_list.append((tagged_label_value[0][i]))
            else:
                if len(curr_list) > 0:
                    components[ConceptType.PASSIVE_NAME.value][tagged_label_key].append(curr_list.copy())
                    curr_list = []
        if len(curr_list) > 0:
            components[ConceptType.PASSIVE_NAME.value][tagged_label_key].append(curr_list.copy())
        curr_list = []
        for i in range(0, len(tagged_label_value[1])):
            if tagged_label_value[1][i] == ConceptType.ACTOR_NAME.value:
                curr_list.append((tagged_label_value[0][i]))
            else:
                if len(curr_list) > 0:
                    components[ConceptType.ACTOR_NAME.value][tagged_label_key].append(curr_list.copy())
                    curr_list = []
        if len(curr_list) > 0:
            components[ConceptType.ACTOR_NAME.value][tagged_label_key].append(curr_list.copy())

    matches = {}
    for label in components[component_type].keys():
        for li in components[component_type][label]:
            for label2 in components[component_type].keys():
                for li2 in components[component_type][label2]:
                    if li == li2:
                        if not ' '.join(li) in matches.keys():
                            matches[' '.join(li)] = {'actions': set(),
                                                     'actor': set(),
                                                     'receiver': set()}
                        if len(components[ConceptType.ACTION_NAME.value][label]) > 0:
                            matches[' '.join(li)]['actions'].add(' '.join(components[ConceptType.ACTION_NAME.value][label][0]))
                        if len(components[ConceptType.ACTOR_NAME.value][label]) > 0:
                            matches[' '.join(li)]['actor'].add(' '.join(components[ConceptType.ACTOR_NAME.value][label][0]))
                        if len(components[ConceptType.PASSIVE_NAME.value][label]) > 0:
                            matches[' '.join(li)]['receiver'].add(' '.join(components[ConceptType.PASSIVE_NAME.value][label][0]))
    trace_bo_lifecycle = {}
    # TODO consider using variants here (unless not only the event label is analyzed)
    # variants = variants_filter.get_variants(ld.get_log_representation())
    label_attribute = XES_NAME
    print('event label is '+label_attribute)
    # TODO rebuild with pandas variant of the event log
    for trace in []: # caseid, case in aug_log.grouped
        case_id = trace.attributes[XES_NAME] # TODO this is misleading, even though it is correct
        trace_bo_lifecycle[case_id] = {}
        for event in trace:
            event_label = preprocess_label(event[label_attribute])
            for li in components[ConceptType.BO_NAME.value][event_label]:
                merged_bo = ' '.join(li)
                if merged_bo not in trace_bo_lifecycle[case_id].keys():
                    trace_bo_lifecycle[case_id][merged_bo] = {ConceptType.ACTION_NAME.value: [], ConceptType.ACTOR_NAME.value: [], ConceptType.PASSIVE_NAME.value: []}
                for li_action in components[ConceptType.ACTION_NAME.value][event_label]:
                    trace_bo_lifecycle[case_id][merged_bo][ConceptType.ACTION_NAME.value].append(' '.join(li_action))
                for li_actor in components[ConceptType.ACTOR_NAME.value][event_label]:
                    trace_bo_lifecycle[case_id][merged_bo][ConceptType.ACTOR_NAME.value].append(' '.join(li_actor))
                for li_receiver in components[ConceptType.PASSIVE_NAME.value][event_label]:
                    trace_bo_lifecycle[case_id][merged_bo][ConceptType.PASSIVE_NAME.value].append(' '.join(li_receiver))
    return matches, trace_bo_lifecycle
