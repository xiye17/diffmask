import itertools
import random
import os
import sys
sys.path.append('.')
from os.path import join
from common.utils import read_json, dump_json, load_bin, dump_to_bin
from collections import OrderedDict
from synth.make_synth_dataset import ENTITY_VOCAB, RELATION_VOCAB

def gen_comp_data():
    num_total = random.choice([3,4])    
    entities = random.sample(ENTITY_VOCAB, num_total)
    relation_pools = random.sample(RELATION_VOCAB, num_total)
    relations = random.choices(relation_pools, k=num_total)
    
    expected_answer = random.choice([True, False])

    # rejection based sampling
    # if expected_answer != (relations[0] == relations[1]):
    #     return None
    
    gt_chains = [entities[0], relations[0], entities[1], relations[1]]

    question = [entities[0], entities[1]]
    gt = 'yes' if (relations[0] == relations[1]) else 'no'
    # print(expected_answer, gt,  relations[0],  relations[1])
    data_chains = [[x,y] for x,y in zip(entities, relations)]
    unique_id = ' '.join(itertools.chain(question, *sorted(data_chains)))

    # prevent overfitting position
    num_fillings = random.choice(list(range(9)))
    data_chains = data_chains + ([['<mask>']] * num_fillings)
    random.shuffle(data_chains)    
    question = ' '.join(question)
    context = ' '.join(itertools.chain(['yes', 'no'], *data_chains))
    # print(question)
    # print(context)
    # print(gt)
    # print(gt_chains)
    # print(unique_id)
    # exit()
    return question, context, gt, gt_chains, unique_id

def get_answer_position(context, answer):
    toks = context.split()
    ans_idx = toks.index(answer)
    ans_start = sum([len(t) for t in toks[:ans_idx]]) + ans_idx
    
    # false_start = context.index(answer)
    # if false_start != ans_start:
    #     global cnt
    #     cnt += 1
    #     print(cnt, context, ' | ', answer)
    #     print(toks, ans_idx)
    #     print(false_start, ' | ', ans_start)
    #     print(context[false_start:false_start + len(answer) + 1], ' | ', context[ans_start:ans_start + len(answer) + 1])

    return ans_start

def make_squad_style_data(question, context, answer, chains, idx):
    answers = [{'answer_start':  get_answer_position(context, answer), 'text': answer}]


    qa0 = {}
    qa0['id'] = f'synth{idx}'
    qa0['question'] = question
    # sanity check    
    qa0['answers'] = answers
    qa0['is_yesno'] = False
    qa0['question_type'] = 'bridge'
    pargraph0 = {'context': context, 'qas': [qa0]}
    data = {'title': ' '.join(chains), 'paragraphs': [pargraph0]}

    return data

def stats_of_data(data):
    unique_ids = [x[4] for x in data]
    answers = [x[2] for x in data]
    num_items = [len(x.split()) for x in unique_ids]
    num_items = [int((x - 2)/2) for x in num_items]
    # print('Num 3', [for n, a in zip(num_items, answers])
    print('Num 3', sum([n == 3 for n in num_items]))
    print('Num 4', sum([n == 4 for n in num_items]))

    print('Num 3 YES', sum([n==3 and a == 'yes' for n, a in zip(num_items, answers)]))
    print('Num 3 NO', sum([n==3 and a == 'no' for n, a in zip(num_items, answers)]))
    print('Num 4 YES', sum([n==4 and a == 'yes' for n, a in zip(num_items, answers)]))
    print('Num 4 NO', sum([n==4 and a == 'no' for n, a in zip(num_items, answers)]))

def generate_comp_dataset(seed=233):
    random.seed(seed)

    train_number = 200000
    dev_number = 10000

    generated = []
    hash_set = set()
    unique_context = set()
    while len(generated) < (train_number + dev_number):
        bundle = gen_comp_data()
        if bundle is None:
            continue
        q, c, a, _ , uid= bundle
        hash_id = uid
        # rejection based
        if a == 'no' and random.random() < 0.6:
            continue
        # print(hash_id)
        # exit()
        if hash_id in hash_set:
            continue
        # u_ctx = hash_id[6:]
        # unique_context.add(u_ctx)
        hash_set.add(hash_id)

        generated.append(bundle)
        # generated.append(('<mask>', c, a, _))
    stats_of_data(generated)
    print(len(unique_context), len(hash_set))
    print(sum([x[2] == 'yes' for x in generated]), sum([x[2] == 'no' for x in generated]))
    # exit()

    generated = [x[:-1] for x in generated]
    train_bundles = generated[:train_number]
    dev_bundles = generated[train_number:]

    train_dataset = [make_squad_style_data(*b, i) for i, b in enumerate(train_bundles)]
    dev_dataset = [make_squad_style_data(*b, i) for i, b in enumerate(dev_bundles)]
    
    # exit()
    output_prefix = 'outputs'
    train_outfile = 'train_comp.json'
    partial_train_outfile = 'partial-train_comp.json'
    dev_outfile = 'dev_comp.json'
    train_outfile = join(output_prefix, train_outfile)
    dev_outfile = join(output_prefix, dev_outfile)
    partial_train_outfile = join(output_prefix, partial_train_outfile)

    partial_train_dataset = {'data': train_dataset[:dev_number], 'version': '1.1'}
    train_dataset = {'data': train_dataset, 'version': '1.1'}
    dev_dataset = {'data': dev_dataset, 'version': '1.1'}
    dump_json(train_dataset, train_outfile)
    dump_json(partial_train_dataset, partial_train_outfile)
    dump_json(dev_dataset, dev_outfile, indent=1)



def generate_balanced_comp_dataset(seed=233):
    random.seed(seed)

    train_number = 200000
    dev_number = 10000

    generated = []
    hash_set = set()
    
    num_per_share = int((train_number + dev_number) / 4)
    
    ans_requirements = ['yes', 'no']
    num_seg_requirements = [3, 4]
    for target_a in ans_requirements:
        for target_n in num_seg_requirements:
            cnt = 0
            while cnt < num_per_share:
                bundle = gen_comp_data()
                if bundle is None:
                    continue
                q, c, a, _ , uid= bundle
                hash_id = uid
                num_item = (len(uid.split()) - 2) // 2
                # rejection based
                if a != target_a or num_item != target_n:
                    continue
                if hash_id in hash_set:
                    continue
                # u_ctx = hash_id[6:]
                # unique_context.add(u_ctx)
                hash_set.add(hash_id)

                generated.append(bundle)
                cnt += 1

    stats_of_data(generated)
    print(sum([x[2] == 'yes' for x in generated]), sum([x[2] == 'no' for x in generated]))

    random.shuffle(generated)
    train_bundles = generated[:train_number]
    dev_bundles = generated[train_number:]
    stats_of_data(dev_bundles)
    train_bundles = [x[:-1] for x in train_bundles]
    dev_bundles = [x[:-1] for x in dev_bundles]
    train_dataset = [make_squad_style_data(*b, i) for i, b in enumerate(train_bundles)]
    dev_dataset = [make_squad_style_data(*b, i) for i, b in enumerate(dev_bundles)]

    output_prefix = 'outputs'
    train_outfile = 'train_comp.json'
    partial_train_outfile = 'partial-train_comp.json'
    dev_outfile = 'dev_comp.json'
    train_outfile = join(output_prefix, train_outfile)
    dev_outfile = join(output_prefix, dev_outfile)
    partial_train_outfile = join(output_prefix, partial_train_outfile)

    partial_train_dataset = {'data': train_dataset[:dev_number], 'version': '1.1'}
    train_dataset = {'data': train_dataset, 'version': '1.1'}
    dev_dataset = {'data': dev_dataset, 'version': '1.1'}
    dump_json(train_dataset, train_outfile)
    dump_json(partial_train_dataset, partial_train_outfile)
    dump_json(dev_dataset, dev_outfile, indent=1)

if __name__ == "__main__":
    # generate_comp_dataset()
    generate_balanced_comp_dataset()
