import requests

from const import ConceptType

system_keywords = ["computer", "information system"]
human_keywords = ["organization", "commercial organization", "social being", "person"]


def get_json(word, cnt=0):
    if word in system_keywords:
        print(word, "sys")
        return ConceptType.SYSTEM_NAME.value
    elif word in human_keywords:
        print(word, "hum")
        return ConceptType.HUMAN_NAME.value
    if cnt > 10:
        return ConceptType.OTHER.value
    obj = requests.get('http://api.conceptnet.io/c/en/'+word).json()
    for edge in obj['edges']:
        print("CN", edge['rel']['label'])
        end = edge['end']['label']
        print(end)
        if edge['rel']['label'] == 'IsA':
            end = edge['end']['label']
            print(end)
            if end != word:
                return get_json(end, cnt+1)
    return ConceptType.OTHER.value


if __name__ == '__main__':
    for word in system_keywords:
        obj = requests.get('http://api.conceptnet.io/c/en/' + word).json()
        print(obj)
        for edge in obj['edges']:
            print("CN", edge['rel']['label'])
            #end = edge['end']['label']
            #print(end)
            if edge['rel']['label'] == 'CapableOf' or edge['rel']['label'] == 'UsedFor':
                end = edge['end']['label']
                print(end)
