TRAING_CASE_1 = {'id': 'physical-commonsense-1426',
'category': 'Status',
'image_id': 'commonsense-physical-commonsense-49',
'question': 'What general conclusion can you draw about this kitchen?',
'choices': ['This is the kitchen of a restaurant',
'The equipment in front has not been cleaned for a long time',
'Someone searched in this kitchen',
'All options are correct'],
'context': '',
'answer': 'D',
'rationale': 'First, the image shows large ovens in a kitchen area that indicates it is a kitchen of a restaurant.\nTherefore, option A is correct.\nSecond, there are grease stains on the front of appliances which are indicative of not being cleaned in a while.\nSo option B is correct answer.\nThird, cabinet doors are opened up throughout the kitchen which shows someone was searching for something.\nSo option C is incorrect.\nTherefore, we can infer that option A, B and C are all correct.\nSo, option "(D) All options are correct" is correct answer.',
'split': 'train',
'image': 'data\\images\\physical-commonsense-1426.png',
'domain': 'commonsense',
'topic': 'physical-commonsense'}


few_shot_demos = '''Question: {}
Options:
A. {}
B. {}
C. {}
D. {}
Answer: {}\n'''.format(TRAING_CASE_1['question'], *TRAING_CASE_1['choices'], TRAING_CASE_1['answer'])

few_shot_cot_demos = '''Question: {}
Options:
A. {}
B. {}
C. {}
D. {}
Let's think step by step.
{}
Answer: {}\n'''.format(TRAING_CASE_1['question'], *TRAING_CASE_1['choices'], TRAING_CASE_1['rationale'] ,TRAING_CASE_1['answer'])

zero_shot_prompt_template = '''Question: {}
Options:
'''

mcot_induct_0 = '''Question: {}
Options:
A. {}
B. {}
C. {}
D. {}
'''.format(TRAING_CASE_1['question'], *TRAING_CASE_1['choices'])


mcot_induct_1 = '''Let's think step by step.\nFirst, the image shows large ovens in a kitchen area that indicates it is a kitchen of a restaurant. Therefore, option A is correct.'''
mcot_induct_2 = '''Second, there are grease stains on the front of appliances which are indicative of not being cleaned in a while. So option B is correct answer.'''
mcot_induct_3 = '''Third, cabinet doors are opened up throughout the kitchen which shows someone was searching for something. So option C is incorrect. Therefore, we can infer that option A, B and C are all correct.'''
mcot_induct_4 = '''Answer: {}\n'''.format(TRAING_CASE_1['answer'])


generation_config = {
    'do_sample': True,
    'temperature': 0.8,
    'top_p': 0.9,
    'max_new_tokens': 128
}
