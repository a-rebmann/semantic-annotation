from const import ConceptType
from model.augmented_log import AugmentedLog
from preprocessing.preprocessor import preprocess_label
import copy


def check_for_missing_bo(ld: AugmentedLog, event_label):
    tagging = ld.get_tags_for_label(event_label)
    return (tagging is not None) and (ConceptType.BO_NAME.value not in tagging[1]), tagging


def check_for_present_a(ld, event_label):
    tagging = ld.get_tags_for_label(event_label)
    return (tagging is not None) and (ConceptType.ACTION_NAME.value in tagging[1]), tagging


def check_for_present_bo(ld, event_label):
    tagging = ld.get_tags_for_label(event_label)
    return (tagging is not None) and (ConceptType.BO_NAME.value in tagging[1]), tagging


def add_bo_to_label(ld, label_with_bo, label_without_bo):
    # TODO assumption that there is only one BO per label
    tagging_with_bo = ld.get_tags_for_label(label_with_bo)
    objects = find_objects(tagging_with_bo[0], tagging_with_bo[1])
    tagging_without_bo = ld.get_tags_for_label(label_without_bo)
    new_tagging = copy.deepcopy(tagging_without_bo)
    for i, tag in enumerate(tagging_without_bo[1]):
        if tag == ConceptType.ACTION_NAME.value and i+1 == len(tagging_without_bo[1]) or \
                tag == ConceptType.ACTION_NAME and tagging_without_bo[1][i + 1] != ConceptType.ACTION_NAME.value:
            j = 1
            for obj in objects:
                new_tagging[0].insert(i+j, obj)
                new_tagging[1].insert(i+j, 'BO')
                j += 1
    ld.add_tagged_label(''.join(new_tagging[0]), new_tagging, label_without_bo)


def find_objects(split, tags):
    return [tok for tok, bo in zip(split, tags) if bo == 'BO']


def augment_labels_with_bo(ld: AugmentedLog):
    """
    Augments labels containing no business object iff - based on the directly follows relation -
    there is a predecessor label that contains a business object
    The assumption is that if there is no business object mentioned but an action is present, the action refers to the
    business object in the previous label with respect to the trace
    :param ld: the LogDescriptor containing all information needed for computation
    :return: the number of labels that have been augmented
    """
    count = 0
    processed_labels = {}
    tuples = {} #TODO get directly follows relations here
    for tup in tuples.keys():
        if tup[0] not in processed_labels.keys():
            processed_labels[tup[0]] = preprocess_label(tup[0])
        if tup[1] not in processed_labels.keys():
            processed_labels[tup[1]] = preprocess_label(tup[1])
        check_bo_present = check_for_present_bo(ld, processed_labels[tup[0]])
        check_bo_absent = check_for_missing_bo(ld, processed_labels[tup[1]])
        check_a_present = check_for_present_a(ld, processed_labels[tup[1]])
        if check_bo_absent[0] and check_bo_present[0] and check_a_present[0]:
            add_bo_to_label(ld, processed_labels[tup[0]], processed_labels[tup[1]])
            count += 1
    return count


