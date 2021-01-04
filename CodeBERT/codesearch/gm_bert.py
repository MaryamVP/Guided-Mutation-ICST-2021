from refactoring_methods import *
from generate_refactoring import *
import numpy
from utils import (compute_metrics, convert_examples_to_features,
                   output_modes, processors)
from bert_extension import *


def refactor(K, file_path):
    for path, d, file_names in os.walk(file_path):
        for filename in file_names:
            if '.java' in filename:
                try:
                    open_file = open(path + '/' + filename, 'r', encoding='ISO-8859-1')
                    code = open_file.read()
                    new_code = generate_adversarial(K, code, filename)
                    wr_path = path + '/' + 'new_' + filename

                    if new_code is not '':
                        l = open(wr_path, 'w')
                        l.write(new_code)

                    else:
                        l = open(wr_path, 'w')
                        l.write(code)

                except Exception:
                    l = open(wr_path, 'w')
                    l.write(code)


def loading_model(model_path):
    model_type = 'roberta'
    tokenizer_name = 'roberta-base'

    config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]

    tokenizer = tokenizer_class.from_pretrained(tokenizer_name, do_lower_case=False)
    model = model_class.from_pretrained(model_path)

    return model, tokenizer


def model_score(model, tokenizer, mode):

    results = evaluate(model, tokenizer, checkpoint=None, prefix='', mode=mode)

    return results['f1']


def mutant_score(mutants_path, tokenizer):
    score = []

    mutant_models = glob.glob(mutants_path + '/*.h5')
    for model in mutant_models:
        score.append(model_score(model, tokenizer, 'test'))

    avg = numpy.average(score)
    return avg


def guided_mutation(generation_number, current_gen, current_coverage, number_of_elites,
                    path, k):

    for current_iteration in range(generation_number):

        sorted_coverage = sorted(range(
            len(current_coverage)), key=lambda i: current_coverage[i], reverse=True)

        for _, list_index in enumerate(sorted_coverage[number_of_elites:]):
            refactor(k, current_gen[list_index])

        current_coverage = mutant_score(path, tokenizer)

    return current_coverage


if __name__ == '__main__':
    k = 1

    file_name = 'valid_orig_50'
    folder_path = '/Users/Vesal/Desktop/CodeBERT-master/data/codesearch/train_valid/java/'
    orig_path = folder_path + file_name

    model_path = 'codesearch/models/java'
    mutants_path = "/Users/Vesal/Desktop/gm_model/models/keras_models/mutants"

    generation_number = 2
    mutation_rate = 0.4

    model, tokenizer = loading_model(model_path)
    current_coverage = evaluate(model, tokenizer, checkpoint=None, prefix='', mode='test')['f1']

    current_gen_len = len(open(orig_path).readlines())
    number_of_elites = int(current_gen_len / 10)

    final_results = guided_mutation(generation_number, orig_path, current_coverage, number_of_elites,
                                    orig_path, k)
    print(final_results)
