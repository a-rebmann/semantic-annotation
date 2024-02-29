
def get_action_framework_info(action_classifier):
    num_per_upper = {}
    for child, upper in action_classifier.child_to_upper_level.items():
        for upp in upper:
            if upp not in num_per_upper:
                num_per_upper[upp] = set()
            num_per_upper[upp].add(child)
    for ke, va in num_per_upper.items():
        print(ke, len(va))