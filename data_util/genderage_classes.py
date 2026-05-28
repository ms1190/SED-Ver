as_strong_train_classes = [
    "child_speech",
    "female_speech",
    "male_speech"
]

as_strong_eval_classes = [
    "child_speech",
    "female_speech",
    "male_speech"
]

genderage_classes = [
    'Child speech, kid speaking',
    'Female speech, woman speaking',
    'Male speech, man speaking'
]

genderage_classes = {
    'Child speech, kid speaking': 0,
    'Female speech, woman speaking': 1,
    'Male speech, man speaking': 2
}

label_names = ['child', 'female', 'male']

'''
indices = []
labels = []

for i, c in enumerate(as_strong_train_classes):
    if c in genderage_classes:
        indices.append(i)
        labels.append(genderage_classes[c])

print(indices)
print(labels)
'''