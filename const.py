import enum

import pm4py.util.xes_constants as xes_const

XES_NAME = xes_const.DEFAULT_NAME_KEY
XES_CASE = 'case:' + xes_const.DEFAULT_TRACEID_KEY
XES_RESOURCE = xes_const.DEFAULT_RESOURCE_KEY
XES_LIFECYCLE = xes_const.DEFAULT_TRANSITION_KEY
XES_TIME = xes_const.DEFAULT_TIMESTAMP_KEY

class ConceptType(enum.Enum):
    BO_NAME = 'BO'
    BO_PROP = 'BO_PROP'
    ACTION_NAME = 'A'
    ACTOR_NAME = 'ACTOR'
    ACTOR_INSTANCE = 'ACTORI'
    PASSIVE_NAME = 'REC'
    PASSIVE_INSTANCE = 'RECI'
    SYSTEM_NAME = "SYS"
    SYSTEM_INSTANCE = "SYSI"
    HUMAN_NAME = "HUM"
    HUMAN_INSTANCE = "HUMI"
    BO_STATUS = 'BOSTATE'
    ACTION_STATUS = 'ASTATE'
    BO_INSTANCE = 'BOI'
    ACTION_INSTANCE = 'AI'
    OTHER = 'X'


type_mapping = {
    ConceptType.BO_NAME.value: 'object:name',
    ConceptType.BO_INSTANCE.value: 'object:instance',
    ConceptType.BO_STATUS.value: 'object:status',
    ConceptType.BO_PROP.value: 'object:property',
    ConceptType.ACTION_NAME.value: 'action:name',
    ConceptType.ACTION_INSTANCE.value: 'action:instance',
    ConceptType.ACTION_STATUS.value: 'action:status',
    ConceptType.ACTOR_NAME.value: 'org:actor:name',
    ConceptType.ACTOR_INSTANCE.value: 'org:actor:instance',
    ConceptType.PASSIVE_NAME.value: 'org:passive:name',
    ConceptType.PASSIVE_INSTANCE.value: 'org:passive:instance',
    ConceptType.SYSTEM_NAME.value: 'org:system:name',
    ConceptType.SYSTEM_INSTANCE.value: 'org:system:instance',
    ConceptType.HUMAN_NAME.value: 'org:human:name',
    ConceptType.HUMAN_INSTANCE.value: 'org:human:instance'
}

class AttributeType(enum.Enum):
    TIMESTAMP = 'Timestamp'  # all datetime attributes and strings that match the datetime patterns
    STRING = 'String'  # string attribute (might contain natural language, but not necessarily)
    TEXT = 'Text'  # string attribute containing natural language, but consists only of single objects
    RICH_TEXT = 'Rich Text'  # string attribute that contains semantically valuable information (multiple semantic
    # components or actions or states)
    NUMERIC = 'Numeric'  # floating point attributes
    INT = 'Integer'  # integer attributes, may be encoded categories
    FLAG = 'Flag'  # a binary attribute (true, false) or (0,1)
    CASE = 'Case Identifier'  # the case identifier of the respective log, there is exactly one, only attribute that
    CASE_ATT = 'Case Attribute'
    # CASE MUST BE KNOWN!
    UNKNOWN = 'Unknown'  # attribute cannot be classified in one of the above categories.


class MatchingMode(enum.Enum):
    EXACT = "exact"
    SEM_SIM = "semsim"

consider_for_tagging = [AttributeType.RICH_TEXT, AttributeType.TEXT]

consider_for_value_classification = [AttributeType.TEXT]

consider_for_label_classification = [AttributeType.TEXT, AttributeType.STRING, AttributeType.UNKNOWN,
                                     AttributeType.INT, AttributeType.FLAG, AttributeType.NUMERIC]

const_to_tag = {"org:role": ConceptType.ACTOR_NAME, "org:group": ConceptType.ACTOR_NAME, XES_CASE: ConceptType.OTHER,
                xes_const.DEFAULT_RESOURCE_KEY: ConceptType.ACTOR_NAME, xes_const.DEFAULT_TRANSITION_KEY: ConceptType.ACTION_STATUS, xes_const.DEFAULT_NAME_KEY: ConceptType.OTHER}

class_labels = ['BO',  'ACTOR',  'BO_PROP',
                'ASTATE', 'BOSTATE', 'A',
                'X']

class_labels_comp = ['BO',  'ACTOR',  'BO_PROP', 'ACTORI', 'BOI',
                'ASTATE', 'BOSTATE', 'A',
                'X']

SCHEMA_ORG = "schema_org"
BPMAI = "bpmai"

CREATE_TERMS = ["create", "register", "receive", "make", "produce", "acquire", "enter", "file", "place"]
# IF ATTRIBUTE VALUES ARE CHECKED THE FOLLOWING WILL BE SIMPLY IGNORED
TERMS_FOR_MISSING = ['MISSING', 'UNDEFINED', 'undefined', 'missing', 'none', 'nan', 'empty', 'empties', 'unknown', 'other', 'others', 'na',
                     'nil', 'null', '', "", ' ', '<unknown>', "0;n/a", "NIL", 'undefined', 'missing', 'none', 'nan', 'empty', 'empties', 'unknown', 'other', 'others', 'na',
                     'nil', 'null', '', ' ']
act = ["activity", "action", "task", "step", "event"]

OC_ACTIVITY = "activity"
OC_TIMESTAMP = "timestamp"

UNIQUE_EVENT_ID = "unique_event_id"

EVAL_MODE = "eval-mode"
MAIN_MODE = "main-mode"

INSTANCE_LEVEL_DATA = 'labels'
ATTRIBUTE_LEVEL_DATA = 'attributes'

RESOURCE_IDX_TO_LABEL = {1: "HUM", 0: "SYS"}

ACTION_IDX_TO_LABEL = {0: "create", 1: "modify", 2: "preserve", 3: "separate", 4: "destroy", 5: "decide", 6: "manage", 7: "combine", 8: "None"}

OCEL_EVENTS = "ocel:events"
OCEL_OMAP = "ocel:omap"
OCEL_VMAP = "ocel:vmap"
OCEL_OVMAP = "ocel:ovmap"
OCEL_TYPE = "ocel:type"
OCEL_ACTIVITY = "ocel:activity"
OCEL_TIMESTAMP = "ocel:timestamp"

N_TO_ONE = "n:1"
N_TO_M = "n:m"
ONE_TO_N = "1:n"

# LABELS FOR THE EVALUATION
OBJ_TYPE = "object:type"
OBJ_INST = "object:instance"
OBJ_PROP = "object:property"
EVENT_ATT = "event:attribute"


CREATION = "creation_based_discovery"
LIFECYCLE = "lifecycle_based_discovery"
CASE_NOTION = "case_notion_based_discovery"

STRICT_ORD = "->"
REVERSE_STRICT_ORD = "<-"
