import itertools
import random
import os
import sys
sys.path.append('.')
from os.path import join
from common.utils import read_json, dump_json, load_bin, dump_to_bin
from collections import OrderedDict

NUM_ENTITY_TYPE = 20
NUM_RELATION_TYPE = 20
ENTITY_VOCAB = [f'E{i}' for i in range(NUM_ENTITY_TYPE)]
RELATION_VOCAB = [f'r{i}' for i in range(NUM_RELATION_TYPE)]
# goal: should consider every piece in the question
# Q: a b C
# C: A a B | B b C 

# attack a
# D c B | B b C

# attack b
# D a E | E c C

# attck C
# D a E | E b F

def make_reasoning_chain(src, r0, bridge, r1, dst):
    return [[src, r0, bridge], [bridge, r1, dst]]

def gen_bridge_data():
    # src 8 bridge 8, dst 2
    involved_entities = random.sample(ENTITY_VOCAB, 18)
    involved_relation = random.sample(RELATION_VOCAB, 4)
    
    src_candidates = involved_entities[:8]
    bridge_candidates = involved_entities[8:16]
    dst_candidates = involved_entities[16:]
    
    relation0_candidates = involved_relation[:2]
    relation1_candidates = involved_relation[2:]

    data_chains = []
    for src, bridge, (r0, r1, dst) in zip(src_candidates, bridge_candidates,
        itertools.product(relation0_candidates, relation1_candidates, dst_candidates)):
        # print(src, r0, bridge, r1, dst)
        data_chains.extend(make_reasoning_chain(src, r0, bridge, r1, dst))
    gt_chains = data_chains[:2]
    
    question = [gt_chains[0][1], gt_chains[1][1], gt_chains[1][2]]
    gt = gt_chains[0][0]
    
    # if random.choice([True, False]):
    #     [x.reverse() for x in data_chains]
    #     question.reverse()
    random.shuffle(data_chains)
    question = ' '.join(question)
    context = ' '.join(itertools.chain(*data_chains))
    gt_chains = [''.join(gt_chains[0]), ''.join(gt_chains[1])]
    # print(question)
    # print(context)
    # print(gt)
    # print(gt_chains)
    return question, context, gt, gt_chains


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
    if answer in ['yes', 'no']:
        answers = [{'answer_start': -1, 'text': answer}]
    else:
        answers = [{'answer_start':  get_answer_position(context, answer), 'text': answer}]


    qa0 = {}
    qa0['id'] = f'synth{idx}'
    qa0['question'] = question
    # sanity check    
    qa0['answers'] = answers
    qa0['is_yesno'] = answer in ['yes', 'no']
    qa0['question_type'] = 'comparison' if answer in ['yes', 'no'] else 'bridge'
    pargraph0 = {'context': context, 'qas': [qa0]}
    data = {'title': ' '.join(chains), 'paragraphs': [pargraph0]}
    return data


def generate_synthetic_dataset(seed=233):
    random.seed(seed)

    train_number = 100000
    dev_number = 10000
    
    generated = []
    hash_set = set()
    while len(generated) < (train_number + dev_number):
        bundle = gen_bridge_data()
        q, c, a, _ = bundle
        hash_id = a + ' ' + q
        if hash_id in hash_set:
            continue
        hash_set.add(hash_id)

        generated.append(bundle)
    
    train_bundles = generated[:train_number]
    dev_bundles = generated[train_number:]

    train_dataset = [make_squad_style_data(*b, i) for i, b in enumerate(train_bundles)]
    dev_dataset = [make_squad_style_data(*b, i) for i, b in enumerate(dev_bundles)]
    
    output_prefix = 'outputs'
    train_outfile = 'train_synth.json'
    partial_train_outfile = 'partial-train_synth.json'
    dev_outfile = 'dev_synth.json'
    train_outfile = join(output_prefix, train_outfile)
    dev_outfile = join(output_prefix, dev_outfile)
    partial_train_outfile = join(output_prefix, partial_train_outfile)

    partial_train_dataset = {'data': train_dataset[:dev_number], 'version': '1.1'}
    train_dataset = {'data': train_dataset, 'version': '1.1'}
    dev_dataset = {'data': dev_dataset, 'version': '1.1'}
    dump_json(train_dataset, train_outfile)
    dump_json(partial_train_dataset, partial_train_outfile)
    dump_json(dev_dataset, dev_outfile, indent=1)



# goal: should consider every piece in the question
# Q: a B
# C: A a B

# attack a
# D b B

# attack B
# D a E

def gen_simple_bridge_data(do_rand=True):
    PROB = 0.5

    involved_entity = random.sample(ENTITY_VOCAB, 6)
    involved_relation = random.sample(RELATION_VOCAB, 3)
    
    src_entity = involved_entity[0]
    dst_entity = involved_entity[1]

    gt_r0 = involved_relation[0]
    gt_chains = [src_entity, gt_r0, dst_entity]

    data_chains = [gt_chains]

    # attack dst
    attack0_src_entity = involved_entity[2]
    attack0_dst_entity = involved_entity[3]
    if do_rand:
        if random.random() < PROB:
            data_chains.append([attack0_src_entity, gt_r0, attack0_dst_entity])
    else:
        data_chains.append([attack0_src_entity, gt_r0, attack0_dst_entity])

    # attack r0, src entity
    attack1_r0 = involved_relation[1]
    attack1_src_entity = involved_entity[4]
    if do_rand:
        if random.random() < PROB:
            data_chains.append([attack1_src_entity , attack1_r0 , dst_entity])
    else:
        data_chains.append([attack1_src_entity , attack1_r0 , dst_entity])

    #  distractor
    dis_src_entity = involved_entity[5]
    dis_dst_entity = attack0_dst_entity
    dis_r0 = attack1_r0
    if do_rand:
        if random.random() < PROB:
            data_chains.append([dis_src_entity, dis_r0, dis_dst_entity])
    else:
        data_chains.append([dis_src_entity, dis_r0, dis_dst_entity])

    # question
    question = [gt_r0, dst_entity]
    gt = src_entity
    unique_id = ' '.join(itertools.chain(question, *sorted(data_chains)))    
    # if random.choice([True, False]):
    #     [x.reverse() for x in data_chains]
    #     question.reverse()

    # prevent overfitting position
    num_fillings = random.choice(list(range(9)))
    data_chains = data_chains + ([['<mask>']] * num_fillings)
    random.shuffle(data_chains)    
    question = ' '.join(question)
    context = ' '.join(itertools.chain(*data_chains))
    return question, context, gt, gt_chains, unique_id

def generate_simple_dataset(seed=233):
    random.seed(seed)

    train_number = 200000
    dev_number = 10000

    generated = []
    hash_set = set()
    unique_context = set()
    while len(generated) < (train_number + dev_number):
        bundle = gen_simple_bridge_data(do_rand=False)
        q, c, a, _ , uid= bundle
        hash_id = uid
        # print(hash_id)
        # exit()
        if hash_id in hash_set:
            continue
        u_ctx = hash_id[6:]
        unique_context.add(u_ctx)
        hash_set.add(hash_id)

        generated.append(bundle[:-1])
        # generated.append(('<mask>', c, a, _))
    print(len(unique_context), len(hash_set))
    train_bundles = generated[:train_number]
    dev_bundles = generated[train_number:]

    train_dataset = [make_squad_style_data(*b, i) for i, b in enumerate(train_bundles)]
    dev_dataset = [make_squad_style_data(*b, i) for i, b in enumerate(dev_bundles)]
    
    output_prefix = 'outputs'
    train_outfile = 'train_simple.json'
    partial_train_outfile = 'partial-train_simple.json'
    dev_outfile = 'dev_simple.json'
    train_outfile = join(output_prefix, train_outfile)
    dev_outfile = join(output_prefix, dev_outfile)
    partial_train_outfile = join(output_prefix, partial_train_outfile)

    partial_train_dataset = {'data': train_dataset[:dev_number], 'version': '1.1'}
    train_dataset = {'data': train_dataset, 'version': '1.1'}
    dev_dataset = {'data': dev_dataset, 'version': '1.1'}
    dump_json(train_dataset, train_outfile)
    dump_json(partial_train_dataset, partial_train_outfile)
    dump_json(dev_dataset, dev_outfile, indent=1)


def sample_sent():
    es = random.sample(ENTITY_VOCAB, 2)
    r = random.choice(RELATION_VOCAB)
    return (es[0], r, es[1])
    
def gen_defective_bridge_data(num_max=3):
    gt_chains = sample_sent()
    num_total = random.choice(list(range(2,num_max + 1)))
    
    data_chains = [gt_chains]
    
    ge0, gr, ge1 = gt_chains
    while len(data_chains) < num_total:
        new_chain = sample_sent()
        if ge0 in new_chain:
            continue
        if gr == new_chain[1] and ge1 == new_chain[2]:
            continue
        data_chains.append(new_chain)

    # question
    question = [gr, ge1]
    gt = ge0
    unique_id = ' '.join(itertools.chain(question, *sorted(data_chains)))    

    # prevent overfitting position
    num_fillings = random.choice(list(range(9)))
    data_chains = data_chains + ([['<mask>']] * num_fillings)
    random.shuffle(data_chains)    
    question = ' '.join(question)
    context = ' '.join(itertools.chain(*data_chains))
    return question, context, gt, gt_chains, unique_id

def generate_defective_dataset(seed=233):
    random.seed(seed)

    train_number = 100000
    dev_number = 10000

    generated = []
    hash_set = set()
    unique_context = set()
    unique_question = set()
    while len(generated) < (train_number + dev_number):
        bundle = gen_defective_bridge_data()
        q, c, a, _ , uid= bundle
        # print(bundles)
        hash_id = uid
        # print(hash_id)
        # exit()
        if hash_id in hash_set:
            continue
        splited_id = hash_id.split(' ')
        u_ctx = ' '.join(splited_id[2:])
        u_q = ' '.join(splited_id[:2])
        unique_context.add(u_ctx)
        unique_question.add(u_q)
        hash_set.add(hash_id)

        generated.append(bundle[:-1])
        # generated.append(('<mask>', c, a, _))
    print(len(unique_context), len(unique_question), len(hash_set))
    train_bundles = generated[:train_number]
    dev_bundles = generated[train_number:]

    train_dataset = [make_squad_style_data(*b, i) for i, b in enumerate(train_bundles)]
    dev_dataset = [make_squad_style_data(*b, i) for i, b in enumerate(dev_bundles)]
    
    output_prefix = 'outputs'
    train_outfile = 'train_simple.json'
    partial_train_outfile = 'partial-train_simple.json'
    dev_outfile = 'dev_simple.json'
    train_outfile = join(output_prefix, train_outfile)
    dev_outfile = join(output_prefix, dev_outfile)
    partial_train_outfile = join(output_prefix, partial_train_outfile)

    partial_train_dataset = {'data': train_dataset[:dev_number], 'version': '1.1'}
    train_dataset = {'data': train_dataset, 'version': '1.1'}
    dev_dataset = {'data': dev_dataset, 'version': '1.1'}
    dump_json(train_dataset, train_outfile)
    dump_json(partial_train_dataset, partial_train_outfile)
    dump_json(dev_dataset, dev_outfile, indent=1)

def generate_simbert_vocab():

    # pad_token="<pad>",
    # unk_token="<unk>",
    # sep_token="<sep>",
    # cls_token="<cls>",
    # mask_token="<mask>",
    tokens = ['<pad>','<cls>','<sep>','<unk>', '<mask>', 'yes', 'no']
    for e in ENTITY_VOCAB:
        tokens.append(e)
    for r in RELATION_VOCAB:
        tokens.append(r)
    
    vocab = OrderedDict()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        vocab[token] = index
    return vocab

if __name__ == "__main__":
    # generate_synthetic_dataset()
    # generate_simple_dataset()
    generate_defective_dataset()
